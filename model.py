import copy
import torch
import numpy as np
import math
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import topK as prData


def cal_squared_l2_distances(data_view):
    '''
    计算平方的欧氏距离
    提示：data_view 的每一行代表一个样本
    '''
    num_samples = data_view.shape[0]

    # 创建一个空的 PyTorch 张量用于存储距离
    dists = torch.zeros((num_samples, num_samples)).cuda()

    # 使用 PyTorch 的广播机制来计算每个样本之间的平方欧氏距离
    for i in range(num_samples):
        dists[i] = torch.sum((data_view - data_view[i]) ** 2, dim=1)

    return dists

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class DIVIDE(torch.nn.Module):
    def __init__(self, n_views, layer_dims, temperature, n_classes, drop_rate=0.5):
        super(DIVIDE, self).__init__()
        self.n_views = n_views
        self.n_classes = n_classes
        self.temperature = temperature

        self.online_encoder = nn.ModuleList([FCN(layer_dims[i], drop_out=drop_rate) for i in range(n_views)])
        self.target_encoder = copy.deepcopy(self.online_encoder)

        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.cross_view_decoder = nn.ModuleList([MLP(layer_dims[i][-1], layer_dims[i][-1]) for i in range(n_views)])

        self.cl = ContrastiveLoss(temperature)
        self.feature_dim = [layer_dims[i][-1] for i in range(n_views)]

    def forward(self, data, momentum, warm_up):
        self._update_target_branch(momentum)
        z_g = torch.tensor([],device='cuda')
        z_q = [self.online_encoder[i](data[i]) for i in range(self.n_views)]
        for z in z_q:
            z_g = torch.cat((z_g, z), 0)  # 沿着第0维拼接
        #print(len(data))
        #print(z_q[0])
        #print(z_q[0])
        #print("Shape of z_q[0]:", z_q[0].shape)
        # print(z_q[1])
        # print("Shape of z_g[1]:", z_q[1].shape)
        # print(z_q[2])
        # print("Shape of z_g[2]:", z_q[2].shape)
        # print(z_g)
        # print("Data type of z_g:", z_g.dtype)
        # print("Shape of z_g:", z_g.shape)
        p = [self.cross_view_decoder[i](z_q[i]) for i in range(self.n_views)]
        #print(p[0])
        z_k = [self.target_encoder[i](data[i]) for i in range(self.n_views)]
        #print(z_k[0])
        #print("Shape of z_k[0]:", z_k[0].shape)
        if warm_up:
            mp = torch.eye(z_q[0].shape[0]).cuda()
            mp = [mp, mp, mp]
        else:
            #随机游走
            mp = [self.kernel_affinity(z_k[i]) for i in range(self.n_views)]
            #print(mp)
            #k阶近邻
            # mp = [self.cal_similiarity_matrix(z_k[i], 10) for i in range(self.n_views)]
        #对比损失
        l_intra = (self.cl(z_q[0], z_k[0], mp[0]) + self.cl(z_q[1], z_k[1], mp[1]) + self.cl(z_q[2], z_k[2], mp[2])) / 3
        l_inter = (self.cl(p[0], z_k[1], mp[1]) + self.cl(p[0], z_k[2], mp[2]) + self.cl(p[1], z_k[0], mp[0]) + self.cl(p[1], z_k[2], mp[2])
                   + self.cl(p[2], z_k[0], mp[0]) + self.cl(p[2], z_k[1], mp[1])) / 6
        #无伪矩阵对比
        # l_intra = (self.cl(z_q[0], z_k[0]) + self.cl(z_q[1], z_k[1]) + self.cl(z_q[2], z_k[2])) / 3
        # l_inter = (self.cl(p[0], z_k[1]) + self.cl(p[0], z_k[2]) + self.cl(p[1], z_k[0]) + self.cl(p[1], z_k[2])
        #            + self.cl(p[2], z_k[0]) + self.cl(p[2], z_k[1])) / 6
        #全局损失
        # l_global = (self.cl(z_g, p[0], mp[0]) + self.cl(z_g, p[1], mp[1]) + self.cl(z_g, p[2], mp[2])) / 3
        # loss = l_inter + l_intra + l_global
        loss = l_inter + l_intra
        return loss

    # @torch.no_grad()
    # #高阶随机游走
    # def kernel_affinity(self, z, temperature=0.1, step: int = 5):
    #     # num_samples = z.shape[0]
    #     z = L2norm(z)
    #     G = (2 - 2 * (z @ z.t())).clamp(min=0.)
    #     # G = cal_squared_l2_distances(z)
    #     # G = torch.zeros((num_samples, num_samples), dtype=torch.float).cuda()
    #     G = torch.exp(-G / temperature)
    #     G = G / G.sum(dim=1, keepdim=True)
    #
    #     G = torch.matrix_power(G, step)
    #     alpha = 0.5
    #     G = torch.eye(G.shape[0]).cuda() * alpha + G * (1 - alpha)
    #     return G


    #使用余弦相似度 进行温度缩放和softmax归一化
    # @torch.no_grad
    # def kernel_affinity(self, z, temperature=0.1, step: int = 5):
    #     z = L2norm(z)
    #
    #     # 使用余弦相似度
    #     G = torch.matmul(z, z.T)
    #     G = G / (torch.norm(z, dim=1, keepdim=True) * torch.norm(z, dim=1, keepdim=True).T)
    #
    #     # 进行温度缩放和softmax归一化
    #     G = torch.softmax(G / temperature, dim=1)
    #
    #     G = torch.matrix_power(G, step)
    #     alpha = 0.5
    #     G = torch.eye(G.shape[0]).cuda() * alpha + G * (1 - alpha)
    #     return G


    # def cl_init(cls, config):
    #     """
    #     Contrastive learning class init function.
    #     """
    #     cls.pooler_type = cls.model_args.pooler_type
    #     cls.pooler = Pooler(cls.model_args.pooler_type)
    #     if cls.model_args.pooler_type == "cls":
    #         cls.mlp = MLPLayer(config)
    #     cls.sim = Similarity(temp=cls.model_args.temp)
    #     cls.init_weights()

    # @torch.no_grad()
    # def kernel_affinity(self, z, temperature=0.1):
    #     # 对输入张量 z 进行 L2 归一化
    #     z = L2norm(z)
    #
    #     # 计算欧氏距离矩阵，并应用温度缩放和指数变换
    #     # G = (2 - 2 * torch.matmul(z, z.T)).clamp(min=0.)  # 计算归一化后的欧氏距离
    #     # G = (2 - 2 * torch.matmul(z, z.T))
    #     # G = torch.exp(-G / temperature)  # 应用温度缩放和指数变换，得到高斯核相似度矩阵
    #     # G = G / G.sum(dim=1, keepdim=True)  # 行归一化
    #     G = torch.matmul(z, z.T)
    #     # 在生成的高斯核相似度矩阵 G 上进行后续计算
    #     # 提取对角线元素并扩展维度，以便与 G 的大小一致
    #     G_diag = torch.diag(G).unsqueeze(1) * torch.ones(1, G.size(0), device=z.device)
    #
    #     # 计算对角线元素与整个相似度矩阵之间的绝对差异
    #     G_diff = torch.abs(G_diag - G)
    #
    #     # 根据设定的阈值（0.7）生成一个二进制权重张量，指示哪些差异小于该阈值
    #     threshold_weights = torch.where(G_diff < 0.7, 1, 0).to(device=z.device)
    #
    #     # 在 G_diff 中加入单位矩阵，以避免后续计算中的零值
    #     G_diff = G_diff + torch.eye(G.size(0), device=z.device)
    #
    #     # 从 G_diff 中获取每行最小的一个元素的索引，找到每个样本与其他样本之间的最小差异
    #     _, topkind = torch.topk(G_diff, 3, dim=1, largest=False)  # 找出最小值的索引
    #     topkind_expand = torch.eye(G.size(0), device=z.device)[topkind]
    #
    #     # 创建一个与 G 形状相同的零张量，用于存储假阳性信息
    #     topkind_false = torch.zeros_like(G, device=z.device)
    #     for i in range(G.size(0)):
    #         for j in range(1):
    #             topkind_false[i] += topkind_expand[i][j]
    #
    #     # 在假阳性张量中加入单位矩阵，以避免后续计算中的零值
    #     topkind_false = topkind_false + torch.eye(G.size(0), device=z.device)
    #
    #     # 生成最终的阈值，通过将阈值权重和假阳性张量相乘得到
    #     topk_threshold = threshold_weights * topkind_false
    #
    #     # 生成一个权重矩阵，指示每个样本的权重，基于对角线元素的归一化
    #     G_diag_exp = torch.exp(G) - torch.diag_embed(torch.diag(torch.exp(G)))
    #     G_weight = torch.ones_like(G, device=z.device) - G_diag_exp / (torch.sum(G_diag_exp, dim=1).unsqueeze(1))
    #
    #     # 计算假阴性权重，通过将阈值与权重矩阵相乘得到
    #     false_negative_weight = topk_threshold * G_weight
    #
    #     # 生成一个与 G 形状相同的张量，表示非假阴性权重
    #     non_false_negative_weight = torch.ones_like(G, device=z.device) - topk_threshold
    #
    #     # 生成动态阈值，通过将假阴性和非假阴性权重相加得到
    #     topk_threshold_dynamic = false_negative_weight + non_false_negative_weight
    #
    #     # 将最终的相似度矩阵与动态阈值相乘，得到调整后的相似度矩阵
    #     adjusted_cos_sim = G * topk_threshold_dynamic
    #
    #     return adjusted_cos_sim  # 返回相似度矩阵

    @torch.no_grad()
    def kernel_affinity(self, z,temperature=0.1):
        # 对输入张量 `z` 进行 L2 归一化
        z = L2norm(z)
        # 计算余弦相似度矩阵 `cos_sim_fix'
        #cos_sim_fix = torch.matmul(z, z.T)
        cos_sim_fix = (2 - 2 * torch.matmul(z, z.T)).clamp(min=0.)
        cos_sim_fix = torch.exp(-cos_sim_fix / temperature)
        cos_sim_fix = cos_sim_fix / cos_sim_fix.sum(dim=1, keepdim=True)  # 行归一化
        # 提取 `cos_sim_fix` 的对角线元素（即自身与自身的相似度），并扩展维度，使其与 `cos_sim_fix` 的大小一致
        cos_sim_diag = torch.diag(cos_sim_fix).unsqueeze(1) * torch.ones(1, cos_sim_fix.size(0), device='cuda')
        # 计算对角线元素与整个相似度矩阵之间的绝对差异
        cos_sim_diff = torch.abs(cos_sim_diag - cos_sim_fix)
        # 根据设定的阈值（0.1）生成一个二进制权重张量，指示哪些差异小于该阈值
        threshold_weights = torch.where(cos_sim_diff < 0.8, 1, 0)
        # 将权重张量移动到 GPU
        threshold_weights = threshold_weights.to(device='cuda')
        # 在 `cos_sim_diff` 中加入单位矩阵，以避免后续计算中的零值
        cos_sim_diff = cos_sim_diff + torch.eye(cos_sim_fix.size(0), device='cuda')
        # 从 `cos_sim_diff` 中获取每行最小的一个元素的索引，找到每个样本与其他样本之间的最小差异
        _, topkind = torch.topk(cos_sim_diff, 3, dim=1, largest=False)  # minimum value
        # 根据 `topkind` 索引生成一个单位矩阵，用于指示前 `topkind` 的位置
        topkind_expand = torch.eye(cos_sim_fix.size(0), device=z.device)[topkind]
        # 将扩展后的张量移动到 GPU
        topkind_expand = topkind_expand.to(device='cuda')
        # 创建一个与 `cos_sim_fix` 形状相同的零张量，用于存储假阳性信息
        topkind_false = torch.zeros_like(cos_sim_fix, device='cuda')
        # 通过循环将 `topkind_expand` 中的值累加到 `topkind_false`，以便为每个实例收集假阳性信息
        for i in range(cos_sim_fix.size(0)):
            for j in range(1):
                topkind_false[i] += topkind_expand[i][j]
        # 在假阳性张量中加入单位矩阵，以避免后续计算中的零值
        topkind_false = topkind_false + torch.eye(cos_sim_fix.size(0), device='cuda')
        # 生成最终的阈值，通过将阈值权重和假阳性张量相乘得到
        topk_threshold = threshold_weights * topkind_false

        # print(topk_threshold)

        # print(topkind_false)
        # 计算 `cos_sim_fix` 的指数并减去对角线元素，以便获取最终的相似度矩阵
        cos_sim_diag = torch.exp(cos_sim_fix) - torch.diag_embed(torch.diag(torch.exp(cos_sim_fix)))
        # 生成一个权重矩阵，指示每个样本的权重，基于对角线元素的归一化
        cos_sim_weight = torch.ones_like(cos_sim_fix, device='cuda') - cos_sim_diag / (
            torch.sum(cos_sim_diag, dim=1).unsqueeze(1))
        # 计算假阴性权重，通过将阈值与权重矩阵相乘得到
        false_negative_weight = topk_threshold * cos_sim_weight
        # 生成一个与 `cos_sim_fix` 形状相同的张量，表示非假阴性权重
        non_false_negative_weight = torch.ones_like(cos_sim_fix, device='cuda') - topk_threshold
        # 生成动态阈值，通过将假阴性和非假阴性权重相加得到
        topk_threshold_dynamic = false_negative_weight + non_false_negative_weight
        # 将最终的相似度矩阵与动态阈值相乘，得到调整后的相似度矩阵
        cos_sim = cos_sim_fix * topk_threshold_dynamic
        #print(cos_sim)

        return cos_sim  # 返回相似度矩阵

    # @torch.no_grad()
    # #k阶近邻
    # def cal_similiarity_matrix(self, data_view, k):
    #     '''
    #     计算相似度矩阵
    #     '''
    #     num_samples = data_view.shape[0]
    #
    #     # 使用 PyTorch 计算平方欧氏距离矩阵
    #     dist = cal_squared_l2_distances(data_view)
    #
    #     # 初始化相似度矩阵为零张量
    #     W = torch.zeros((num_samples, num_samples), dtype=torch.float).cuda()
    #
    #     # 按升序对距离进行排序并获取索引
    #     idx_set = torch.argsort(dist, dim=1)
    #
    #     # 计算相似度矩阵
    #     for i in range(num_samples):
    #         # 获取样本 i 的最近邻索引集，去掉第一个（自身），选取 k+1 个邻居
    #         idx_sub_set = idx_set[i, 1:(k + 2)]
    #
    #         # 获取样本 i 与邻居的距离
    #         di = dist[i, idx_sub_set]
    #
    #         # 计算相似度权重，避免除以 0
    #         W[i, idx_sub_set] = (di[k] - di) / (
    #                     di[k] - torch.mean(di[0:(k - 1)]) + torch.tensor(math.e).cuda())
    #
    #     # 确保相似度矩阵是对称的
    #     W = (W + W.T) / 2
    #
    #     return W

    @torch.no_grad()
    def _update_target_branch(self, momentum):
        for i in range(self.n_views):
            for param_k, param_q in zip(self.online_encoder[i].parameters(), self.target_encoder[i].parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1 - momentum)
    def _update_target_branch(self, momentum):
        for i in range(self.n_views):
            for param_o, param_t in zip(self.online_encoder[i].parameters(), self.target_encoder[i].parameters()):
                param_t.data = param_t.data * momentum + param_o.data * (1 - momentum)

    @torch.no_grad()
    def extract_feature(self, data, mask):
        N = data[0].shape[0]
        z = [torch.zeros(N, self.feature_dim[i]).cuda() for i in range(self.n_views)]
        for i in range(self.n_views):
            z[i][mask[:, i]] = self.target_encoder[i](data[i][mask[:, i]])

        for i in range(self.n_views):
            z[i][~mask[:, i]] = self.cross_view_decoder[1 - i](z[1 - i][~mask[:, i]])

        z = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        z = [L2norm(z[i]) for i in range(self.n_views)]

        return z


import torch.nn as nn
import torch.nn.functional as F
L2norm = nn.functional.normalize

class FCN(nn.Module):
    def __init__(self, dim_layer=None, norm_layer=None, act_layer=None, drop_out=0.0, norm_last_layer=True):
        super(FCN, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm1d
        layers = []
        for i in range(1, len(dim_layer) - 1):
            layers.append(nn.Linear(dim_layer[i - 1], dim_layer[i], bias=False))
            layers.append(norm_layer(dim_layer[i]))
            layers.append(act_layer())
            if drop_out != 0.0 and i != len(dim_layer) - 2:
                layers.append(nn.Dropout(drop_out))

        if norm_last_layer:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=False))
            layers.append(nn.BatchNorm1d(dim_layer[-1], affine=False))
        else:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=True))

        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        return self.ffn(x)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out=None, hidden_ratio=4.0, act_layer=None):
        super(MLP, self).__init__()
        dim_out = dim_out or dim_in
        dim_hidden = int(dim_in * hidden_ratio)
        act_layer = act_layer or nn.ReLU
        self.mlp = nn.Sequential(nn.Linear(dim_in, dim_hidden),
                                 act_layer(),
                                 nn.Linear(dim_hidden, dim_out))

    def forward(self, x):
        x = self.mlp(x)
        return x


# class ContrastiveLoss(nn.Module):
#     def __init__(self, temperature=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.temperature = temperature
#
#     def forward(self, x_q, x_k, mask_pos=None):
#         x_q = L2norm(x_q)
#         x_k = L2norm(x_k)
#         N = x_q.shape[0]
#         if mask_pos is None:
#             mask_pos = torch.eye(N).cuda()
#         similarity = torch.div(torch.matmul(x_q, x_k.T), self.temperature)
#         log_similarity = F.log_softmax(similarity, dim=1)
#         nll_loss = -log_similarity * mask_pos / mask_pos.sum(dim=1, keepdim=True)
#         loss = nll_loss.mean()
#         return loss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x_q, x_k, mask_pos=None):
        x_q = L2norm(x_q)
        x_k = L2norm(x_k)
        N = x_q.shape[0]
        if mask_pos is None:
            mask_pos = torch.eye(N).cuda()
        similarity = torch.div(torch.matmul(x_q, x_k.T), self.temperature)
        similarity = -torch.log(torch.softmax(similarity, dim=1))
        nll_loss = similarity * mask_pos / mask_pos.sum(dim=1, keepdim=True)
        loss = nll_loss.mean()
        return loss