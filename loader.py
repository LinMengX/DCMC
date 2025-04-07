import os.path
import torch
from torch.utils.data import ConcatDataset, Dataset
import scipy.io as sio
from scipy import sparse

import sklearn.preprocessing as skp
import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder

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

    sampleMean = np.mean(input, axis=1).reshape(input.shape[0], 1)


    sampleStd = np.std(input, axis=1).reshape(input.shape[0], 1)


    input = (input - sampleMean) / sampleStd


    sampleNorm = np.linalg.norm(input, axis=1).reshape(input.shape[0], 1)


    if unitNorm:

        input = input / sampleNorm


    return input
data_X = []
label_y = None
data_path = "./data"
view_names = ['exp', 'mirna', 'methy']  
cancerName = 'aml'
fileName = cancerName + '.mat'
filePath = data_path + '/' + fileName
mat = _check_keys(sio.loadmat(filePath,struct_as_record=False, squeeze_me=True))

for vname in view_names:

    curData = np.array(list(mat[vname].values()))

    curData = normalization_v2(curData)

    data_X.append(torch.from_numpy(curData).float())

labels = list(mat[view_names[0]].keys())
print(f'Number of views: {len(data_X)}\nNumber of samples: {len(labels)}')

