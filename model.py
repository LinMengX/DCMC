import copy
import torch
import numpy as np
import math
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import topK as prData


def cal_squared_l2_distances(data_view):
  
    num_samples = data_view.shape[0]

    dists = torch.zeros((num_samples, num_samples)).cuda()

    for i in range(num_samples):
        dists[i] = torch.sum((data_view - data_view[i]) ** 2, dim=1)

    return dists

class MLPLayer(nn.Module):
   

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

class DCMC(torch.nn.Module):
    def __init__(self, n_views, layer_dims, temperature, n_classes, drop_rate=0.5):
        super(DCMC, self).__init__()
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
            z_g = torch.cat((z_g, z), 0) 
        p = [self.cross_view_decoder[i](z_q[i]) for i in range(self.n_views)]
        z_k = [self.target_encoder[i](data[i]) for i in range(self.n_views)]
        if warm_up:
            mp = torch.eye(z_q[0].shape[0]).cuda()
            mp = [mp, mp, mp]
        else:
            mp = [self.cal_similiarity_matrix(z_k[i]) for i in range(self.n_views)]
            # mp = [self.cal_similiarity_matrix(z_k[i], 10) for i in range(self.n_views)]

        l_intra = (self.cl(z_q[0], z_k[0], mp[0]) + self.cl(z_q[1], z_k[1], mp[1]) + self.cl(z_q[2], z_k[2], mp[2])) / 3
        l_inter = (self.cl(p[0], z_k[1], mp[1]) + self.cl(p[0], z_k[2], mp[2]) + self.cl(p[1], z_k[0], mp[0]) + self.cl(p[1], z_k[2], mp[2])
                   + self.cl(p[2], z_k[0], mp[0]) + self.cl(p[2], z_k[1], mp[1])) / 6

        loss = l_inter + l_intra
        return loss


    @torch.no_grad()
    def cal_similiarity_matrix(self, z,temperature=0.1):
        z = L2norm(z)
        #cos_sim_fix = torch.matmul(z, z.T)
        cos_sim_fix = (2 - 2 * torch.matmul(z, z.T)).clamp(min=0.)
        cos_sim_fix = torch.exp(-cos_sim_fix / temperature)
        cos_sim_fix = cos_sim_fix / cos_sim_fix.sum(dim=1, keepdim=True)  
        cos_sim_diag = torch.diag(cos_sim_fix).unsqueeze(1) * torch.ones(1, cos_sim_fix.size(0), device='cuda')
        
        cos_sim_diff = torch.abs(cos_sim_diag - cos_sim_fix)
        
        threshold_weights = torch.where(cos_sim_diff < 0.8, 1, 0)
      
        threshold_weights = threshold_weights.to(device='cuda')
       
        cos_sim_diff = cos_sim_diff + torch.eye(cos_sim_fix.size(0), device='cuda')
        
        _, topkind = torch.topk(cos_sim_diff, 3, dim=1, largest=False)  # minimum value
      
        topkind_expand = torch.eye(cos_sim_fix.size(0), device=z.device)[topkind]
       
        topkind_expand = topkind_expand.to(device='cuda')
        
        topkind_false = torch.zeros_like(cos_sim_fix, device='cuda')
        
        for i in range(cos_sim_fix.size(0)):
            for j in range(1):
                topkind_false[i] += topkind_expand[i][j]
        
        topkind_false = topkind_false + torch.eye(cos_sim_fix.size(0), device='cuda')
       
        topk_threshold = threshold_weights * topkind_false

        cos_sim_diag = torch.exp(cos_sim_fix) - torch.diag_embed(torch.diag(torch.exp(cos_sim_fix)))
        
        cos_sim_weight = torch.ones_like(cos_sim_fix, device='cuda') - cos_sim_diag / (
            torch.sum(cos_sim_diag, dim=1).unsqueeze(1))
       
        false_negative_weight = topk_threshold * cos_sim_weight
        
        non_false_negative_weight = torch.ones_like(cos_sim_fix, device='cuda') - topk_threshold
        
        topk_threshold_dynamic = false_negative_weight + non_false_negative_weight
        
        cos_sim = cos_sim_fix * topk_threshold_dynamic
       
        return cos_sim  

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