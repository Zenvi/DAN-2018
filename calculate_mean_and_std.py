'''
This script calculates the mean and std values given source and target names.
Usage:
Manually modify the 'Parameters' section and run the code
'''

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

import utils

def get_mean_std_value(loader_list):
    '''
    求数据集的均值和标准差
    :param loader:
    :return:
    '''
    data_sum,data_squared_sum,num_batches = 0,0,0
    for loader in loader_list:
        for data,_,_ in loader:
            # data: [batch_size,channels,height,width]
            # 计算dim=0,2,3维度的均值和，dim=1为通道数量，不用参与计算
            data_sum += torch.mean(data,dim=[0,2,3])    # [batch_size,channels,height,width]
            # 计算dim=0,2,3维度的平方均值和，dim=1为通道数量，不用参与计算
            data_squared_sum += torch.mean(data**2,dim=[0,2,3])  # [batch_size,channels,height,width]
            # 统计batch的数量
            num_batches += 1
    # 计算均值
    mean = data_sum/num_batches
    # 计算标准差
    std = (data_squared_sum/num_batches - mean**2)**0.5
    return mean,std

# Parameters
data = 'Office31'
root = 'running_outputs/data/office31'  # root path of dataset
source = ['A']  # 2817
target = ['W']  # 795
batch_size = 32
workers = 0

train_transform = T.Compose([
            T.Resize([256, 256]),
            T.CenterCrop(224),
            T.ToTensor()
        ])
val_transform = T.Compose([
            T.Resize([256, 256]),
            T.CenterCrop(224),
            T.ToTensor()
        ])


train_source_dataset, \
train_target_dataset, \
val_dataset, \
test_dataset, \
num_classes, \
class_names = utils.get_dataset(data, root, source, target, train_transform, val_transform)

train_source_loader = DataLoader(train_source_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=workers,
                                 drop_last=True)

train_target_loader = DataLoader(train_target_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=workers,
                                 drop_last=True)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

print('Source Domain {}:'.format(source), get_mean_std_value([train_source_loader]))
print('Target Domain {}:'.format(target), get_mean_std_value([train_target_loader]))
print('Source & Target Joint:', get_mean_std_value([train_source_loader, train_target_loader]))