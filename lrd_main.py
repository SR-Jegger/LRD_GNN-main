import yaml
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Actor, WikipediaNetwork, WebKB, LINKXDataset, WikiCS
import math
import numpy as np
from yaml import SafeLoader
from base_area.utils import data_splits
from model import RPCA_model


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Texas')
parser.add_argument('--dropout', type=int, default=0.2)
parser.add_argument('--lr', type=int, default=0.01)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--weight', type=int, default=0.001)
parser.add_argument('--batchnorm', type=bool, default=False)
parser.add_argument('--config', type=str, default='config.yaml')
args = parser.parse_args()


dataset_name = args.dataset

if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root='./dataset', name=dataset_name)
elif dataset_name in ['chameleon', 'squirrel', 'CS']:
    dataset = WikipediaNetwork(root='./dataset', name=dataset_name)
elif dataset_name in ['Photo', 'Computers']:
    dataset = Amazon(root='./dataset', name=dataset_name)
elif dataset_name in ['Cornell', 'Wisconsin','Texas']:
    dataset = WebKB(root='./dataset', name=dataset_name)
elif dataset_name in ['Actor']:
    dataset = Actor(root='./dataset/Actor')
elif dataset_name in ['wikics']:
    dataset = WikiCS(root='./dataset/wikics')

data = dataset[0]
num_nodes = data.num_nodes
num_features = data.num_features
edge_index = data.edge_index
nclasses = dataset.num_classes

config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
filename = config['filename']

data_path='./my_data/' + dataset_name +'/'+ filename
np.set_printoptions(threshold=np.inf)
features = np.load(data_path,allow_pickle=True)
features = torch.from_numpy(features)
features = features.to(torch.float32)
data.x = features


data.train_mask, data.val_mask, data.test_mask  = data_splits(dataset_name, num_nodes)


###################hyperparameters
dropout = config['dropout']
hidden_dim = config['hidden_dim']
weight_decay = config['weight_decay']
lr = config['learning_rate']
batchnorm = config['batchnorm']
#####################

patience = 200
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model, data = RPCA_model(num_features, nclasses, hidden_dim, dropout, batchnorm).to(device), data.to(device)
optimizer = torch.optim.Adam([
    #dict(params=model.reg_params, weight_decay=weight_decay1),
    dict(params=model.non_reg_params, weight_decay=weight_decay)
], lr=lr)
data = data.to(device)


def train():
    model.train()
    optimizer.zero_grad()
    loss_train = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
    # log = model(data)
    loss_train.backward()
    optimizer.step()
    return loss_train.item()

@torch.no_grad()
def test():
    model.eval()
    logits = model(data)
    loss_val = F.nll_loss(logits[data.val_mask], data.y[data.val_mask]).item()
    for _, mask in data('test_mask'):
        pred = logits[mask].max(1)[1]
        accs = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return loss_val, accs

end_test_f1_list=[]
for time in range(10):
    best_val_loss = 9999999
    test_acc = 0
    bad_counter = 0
    best_epoch = 0
    for epoch in range(1, 1300):
        loss_tra = train()
        loss_val,acc_test_tmp = test()
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            test_acc = acc_test_tmp
            bad_counter = 0
            best_epoch = epoch
        else:
            bad_counter+=1

        log = 'Epoch: {:03d}, Train loss: {:.4f}, Val loss: {:.4f}, Test acc: {:}' #.4f
        #print(log.format(epoch, loss_tra, loss_val, test_acc))
        print(log.format(epoch, loss_tra, loss_val, acc_test_tmp))
        if bad_counter == patience:
            break

    end_test_f1_list.append(test_acc)
    log = 'best Epoch: {:03d}, Val loss: {:.4f}, Test acc: {:}'  #.4f
    print(log.format(best_epoch, best_val_loss, test_acc))


print(end_test_f1_list)
res_list = np.asarray(end_test_f1_list)
last_acc=np.mean(res_list)
"""@nni.report_final_result(last_acc)"""
print(last_acc)
print('Average：',np.mean(res_list))
print('Standard deviation：',np.std(res_list))


