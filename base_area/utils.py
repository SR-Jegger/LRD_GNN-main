import argparse
import torch
from torch.utils.data import random_split
import numpy as np

def data_splits(dataset_name, num_nodes):
    if dataset_name in ['chameleon', 'squirrel', 'Texas','Cornell', 'Wisconsin', 'Actor']:
        # train_mask = torch.tensor([num_nodes], dtype=torch.bool)
        # val_mask = torch.tensor([num_nodes], dtype=torch.bool)
        # test_mask = torch.tensor([num_nodes], dtype=torch.bool)
        splits_path = './dataset_splits/'
        data_splits = np.load(splits_path+dataset_name+'_split_0.6_0.2_6.npz')
        #data_splits = np.load(splits_path + dataset_name + '_split_0.48_0.2.npz')
        train_mask = torch.from_numpy(data_splits['train_mask'])
        val_mask = torch.from_numpy(data_splits['val_mask'])
        test_mask = torch.from_numpy(data_splits['test_mask'])
    #elif dataset_name in ['wikics', 'Computers', 'Photo']:
    else:
        train_mask, val_mask, test_mask = splits(num_nodes)
    return  train_mask, val_mask, test_mask



# # todo 自己直接划分数据集
def splits(num_nodes):
    train_mask = torch.zeros(num_nodes).bool()
    val_mask = torch.zeros(num_nodes).bool()
    test_mask = torch.zeros(num_nodes).bool()
    num_train = int(0.1*num_nodes)
    num_test = int(0.8*num_nodes)
    num_val = num_nodes-num_train-num_test
    #dataset = range(data.num_nodes)
    train_dataset, val_dataset, test_dataset = random_split(
        range(num_nodes),
        lengths=[num_train, num_val, num_test],
        generator=torch.Generator()
    )
    train_mask[list(train_dataset)] = True
    val_mask[list(val_dataset)] = True
    test_mask[list(test_dataset)] = True
    # np.savez("./dataset_splits/Photo_split_0.1_0.8.npz_split.npz",train_mask=train_mask,val_mask=val_mask,test_mask=test_mask)
    return  train_mask, val_mask, test_mask
