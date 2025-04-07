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

        for theta_a, theta_b in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            theta_b.data.copy_(theta_a.data)  # initialize
            theta_b.requires_grad = False  # not updated by gradient

        self.cross_view_decoder = nn.ModuleList([MLP(layer_dims[i][-1], layer_dims[i][-1]) for i in range(n_views)])

        self.cl = ContrastiveLoss(temperature)
        self.feature_dim = [layer_dims[i][-1] for i in range(n_views)]

    def forward(self, data, momentum, warm_up):
        self._update_target_branch(momentum)

        f_a = [self.online_encoder[i](data[i]) for i in range(self.n_views)]

        Q = [self.cross_view_decoder[i](f_a[i]) for i in range(self.n_views)]
        f_b = [self.target_encoder[i](data[i]) for i in range(self.n_views)]

        if warm_up:
            ne = torch.eye(f_a[0].shape[0]).cuda()
            ne = [ne, ne, ne]
        else:
            ne = [self.cal_similiarity_matrix(f_b[i]) for i in range(self.n_views)]
            # ne = [self.cal_similiarity_matrix(f_b[i], 10) for i in range(self.n_views)]

        l_intra = (self.cl(f_a[0], f_b[0], ne[0]) + self.cl(f_a[1], f_b[1], ne[1]) + self.cl(f_a[2], f_b[2], ne[2])) / 3
        l_inter = (self.cl(Q[0], f_b[1], ne[1]) + self.cl(Q[0], f_b[2], ne[2]) + self.cl(Q[1], f_b[0], ne[0]) +
                   self.cl(Q[1], f_b[2], ne[2]) + self.cl(Q[2], f_b[0], ne[0]) + self.cl(Q[2], f_b[1], ne[1])) / 6

        loss = l_inter + l_intra
        return loss

    @torch.no_grad()
    def cal_similiarity_matrix(self, features, temperature=0.1):
        features = L2norm(features)  # L2 normalization

        euclidean_sim = (2 - 2 * torch.matmul(features, features.T)).clamp(min=0.)
        sim_matrix = torch.exp(-euclidean_sim / temperature)
        sim_matrix = sim_matrix / sim_matrix.sum(dim=1, keepdim=True)

        self_sim_column = torch.diag(sim_matrix).unsqueeze(1)
        self_sim_matrix = self_sim_column * torch.ones(1, sim_matrix.size(0), device='cuda')

        sim_difference = torch.abs(self_sim_matrix - sim_matrix)

        threshold_filter = torch.where(sim_difference < 0.7, 1, 0).to(device='cuda')

        sim_difference = sim_difference + torch.eye(sim_matrix.size(0), device='cuda')

        _, topk_indices = torch.topk(sim_difference, 3, dim=1, largest=False)
        topk_selector = torch.eye(sim_matrix.size(0), device='cuda')[topk_indices]

        possible_fn = torch.zeros_like(sim_matrix, device='cuda')
        for i in range(sim_matrix.size(0)):
            for j in range(1):  # j=0
                possible_fn[i] += topk_selector[i][j]

        possible_fn = possible_fn + torch.eye(sim_matrix.size(0), device='cuda')

        selected_fn = threshold_filter * possible_fn

        sim_matrix_exp = torch.exp(sim_matrix) - torch.diag_embed(torch.diag(torch.exp(sim_matrix)))
        weight_matrix = 1 - sim_matrix_exp / sim_matrix_exp.sum(dim=1, keepdim=True)

        fn_weight = selected_fn * weight_matrix
        rest_weight = 1 - selected_fn

        adaptive_weight = fn_weight + rest_weight
        weighted_sim_matrix = sim_matrix * adaptive_weight

        return weighted_sim_matrix

    @torch.no_grad()
    def _update_target_branch(self, momentum):
        for i in range(self.n_views):
            for theta_b, theta_a in zip(self.online_encoder[i].parameters(), self.target_encoder[i].parameters()):
                theta_b.data = theta_b.data * momentum + theta_a.data * (1 - momentum)

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