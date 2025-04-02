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
    """
    检查字典中的条目是否为 mat 对象。如果是，则调用 _todict 将其转换为嵌套字典。
    """
    for key in dict:
        # 检查字典中每个条目的类型是否为 sio.matlab.mio5_params.mat_struct 类型
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            # 如果是 mat_struct 类型，则调用 _todict 函数将其转换为嵌套字典
            dict[key] = _todict(dict[key])
            # 打印转换后的字典条目
            #print("打印转换后的字典条目")
            #print(dict[key])
    # 返回转换后的字典
    return dict
def _todict(matobj):
    """
    一个递归函数，用于从 mat 对象构建嵌套字典
    """
    dict = {}
    # 创建一个空字典用于存储转换后的键值对
    for strg in matobj._fieldnames:
        # 遍历 mat 对象的所有字段名
        elem = matobj.__dict__[strg]
        # 通过字段名访问 mat 对象的属性
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            # 如果属性是 mat_struct 类型，则递归调用 _todict 将其转换为嵌套字典
            dict[strg] = _todict(elem)
        else:
            # 否则，直接将属性值添加到字典中
            dict[strg] = elem
    # 返回构建好的嵌套字典
    #print("构建好的嵌套字典")
    #print(dict)
    return dict
def normalization_v2(input, unitNorm=True):
    """ mean 0 std 1, with unit norm """

    # 计算每个样本（行）的均值，并将其形状调整为 (n, 1)
    sampleMean = np.mean(input, axis=1).reshape(input.shape[0], 1)

    # 计算每个样本（行）的标准差，并将其形状调整为 (n, 1)
    sampleStd = np.std(input, axis=1).reshape(input.shape[0], 1)

    # 对输入矩阵进行均值和标准差归一化
    input = (input - sampleMean) / sampleStd

    # 计算每个样本（行）的范数，并将其形状调整为 (n, 1)
    sampleNorm = np.linalg.norm(input, axis=1).reshape(input.shape[0], 1)

    # 如果需要单位范数归一化
    if unitNorm:
        # 将输入矩阵除以每个样本的范数，实现单位范数归一化
        input = input / sampleNorm

    # 返回归一化后的输入矩阵
    return input
data_X = []
label_y = None
data_path = "./data"
view_names = ['exp', 'mirna', 'methy']  # 视图名称列表
cancerName = 'aml'
fileName = cancerName + '.mat'
filePath = data_path + '/' + fileName
mat = _check_keys(sio.loadmat(filePath,struct_as_record=False, squeeze_me=True))
#print(mat)
for vname in view_names:
    # 获取当前视图的数据并转换为 numpy 数组
    curData = np.array(list(mat[vname].values()))
    # print("获取当前视图的数据并转换为 numpy 数组")
    print("标准化前")
    print(curData)
    #两种标准化方法
    curData = normalization_v2(curData)
    #curData = skp.normalize(curData)
    print("标准化后")
    print(curData)
    # 对数据进行标准化并转换为 PyTorch 张量，类型为 float
    data_X.append(torch.from_numpy(curData).float())

# data_X.append(mat['mirna'].astype('float32'))
# data_X.append(mat['methy'].astype('float32'))
#print(data_X)
#label_y = np.squeeze(mat['Y'])
labels = list(mat[view_names[0]].keys())
print(f'Number of views: {len(data_X)}\nNumber of samples: {len(labels)}')
# n_sample = data_X[0].shape[0]
