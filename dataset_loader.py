import os.path
import torch
from torch.utils.data import ConcatDataset, Dataset
import scipy.io as sio
from scipy import sparse

import sklearn.preprocessing as skp

def _check_keys(dict):
   
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            
            dict[key] = _todict(dict[key])
          
    return dict
def _todict(matobj):
    
    dict = {}
    
    for strg in matobj._fieldnames:
        
        elem = matobj.__dict__[strg]
      
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            
            dict[strg] = _todict(elem)
        else:
            
            dict[strg] = elem
    return dict
def normalization_v2(input, unitNorm=True):
    """ mean 0 std 1, with unit norm """

    sampleMean = np.mean(input, axis=1).reshape(input.shape[0], 1)

    sampleStd = np.std(input, axis=1).reshape(input.shape[0], 1)

    input = (input - sampleMean) / sampleStd

    sampleNorm = np.linalg.norm(input, axis=1).reshape(input.shape[0], 1)

    if unitNorm:
        input = input / sampleNorm
    return input
def load_mat(args):
    data_X = []
    label_y = None
    view_names = ['exp', 'mirna', 'methy']  
    cancerName = args.dataset
    fileName = cancerName + '.mat'
    filePath = args.data_path + '/' + fileName
    mat = _check_keys(sio.loadmat(filePath, struct_as_record=False, squeeze_me=True))
    for vname in view_names:
        curData = np.array(list(mat[vname].values()))
       
        curData = normalization_v2(curData)
        # curData = skp.normalize(curData)
        
        data_X.append(torch.from_numpy(curData).float())
    labels = list(mat[view_names[0]].keys())
    args.n_sample = len(labels)
    print(f'Number of views: {len(data_X)}\nNumber of samples: {len(labels)}')
    return data_X, labels


def load_dataset(args):
    data, targets = load_mat(args)
    #dataset = IncompleteMultiviewDataset(args.n_views, data, targets, args.missing_rate)
    dataset = IncompleteMultiviewDataset(args.n_views, data, args.missing_rate)
    return dataset


class MultiviewDataset(torch.utils.data.Dataset):
    def __init__(self, n_views, data_X, label_y):
        super(MultiviewDataset, self).__init__()
        self.n_views = n_views
        self.data = data_X
       

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        data = []
        for i in range(self.n_views):
            #data.append(torch.tensor(self.data[i][idx].astype('float32')))
            data.append(self.data[i][idx].float())
       
        return idx, data

import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder


class IncompleteMultiviewDataset(torch.utils.data.Dataset):
    def __init__(self, n_views, data_X, missing_rate):
        super(IncompleteMultiviewDataset, self).__init__()
        self.n_views = n_views
        self.data = data_X
       

        self.missing_mask = torch.from_numpy(self._get_mask(n_views, self.data[0].shape[0], missing_rate)).bool()

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        data = []
        for i in range(self.n_views):
           
            data.append(self.data[i][idx].float())
      
        mask = self.missing_mask[idx]
        
        return idx, data, mask

    @staticmethod
    def _get_mask(view_num, alldata_len, missing_rate):
        
        full_matrix = np.ones((int(alldata_len * (1 - missing_rate)), view_num))

        alldata_len = alldata_len - int(alldata_len * (1 - missing_rate))
        missing_rate = 0.5
        if alldata_len != 0:
            one_rate = 1.0 - missing_rate
            if one_rate <= (1 / view_num):
                enc = OneHotEncoder()
                view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
                full_matrix = np.concatenate([view_preserve, full_matrix], axis=0)
                choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
                matrix = full_matrix[choice]
                return matrix
            error = 1
            if one_rate == 1:
                matrix = randint(1, 2, size=(alldata_len, view_num))
                full_matrix = np.concatenate([matrix, full_matrix], axis=0)
                choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
                matrix = full_matrix[choice]
                return matrix
            while error >= 0.005:
                enc = OneHotEncoder()
                view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
                one_num = view_num * alldata_len * one_rate - alldata_len
                ratio = one_num / (view_num * alldata_len)
                matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
                a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
                one_num_iter = one_num / (1 - a / one_num)
                ratio = one_num_iter / (view_num * alldata_len)
                matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
                matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
                ratio = np.sum(matrix) / (view_num * alldata_len)
                error = abs(one_rate - ratio)
            full_matrix = np.concatenate([matrix, full_matrix], axis=0)
        choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
        matrix = full_matrix[choice]
        return matrix


class IncompleteDatasetSampler:
    def __init__(self, dataset: Dataset, seed: int = 0, drop_last: bool = False) -> None:
        self.dataset = dataset
        self.epoch = 0
        self.drop_last = drop_last
        self.seed = seed
        self.compelte_idx = torch.where(self.dataset.missing_mask.sum(dim=1) == self.dataset.n_views)[0]
        self.num_samples = self.compelte_idx.shape[0]

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = torch.randperm(self.num_samples, generator=g).tolist()

        indices = self.compelte_idx[indices].tolist()

        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch
