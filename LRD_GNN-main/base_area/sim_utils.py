import networkx as nx
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB, Actor, LINKXDataset, Amazon, Coauthor, WikiCS
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
from TRPCA_torch import TRPCA



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


dataset_name = 'chameleon'

homo_set = ['Cora', 'CiteSeer', 'PubMed', 'Photo', 'Computers', 'wikics']
heter_set = ['chameleon', 'squirrel', 'Actor', 'Cornell', 'Wisconsin','Texas']
if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root='../dataset', name=dataset_name)
elif dataset_name in ['chameleon', 'squirrel']:
    dataset = WikipediaNetwork(root='../dataset', name=dataset_name)
elif dataset_name in ['Cornell', 'Wisconsin','Texas']:
    dataset = WebKB(root='../dataset', name=dataset_name)
elif dataset_name in ['Actor']:
    dataset = Actor(root='../dataset/Actor')
elif dataset_name in ['Photo', 'Computers']:
    dataset = Amazon(root='../dataset', name=dataset_name)
elif dataset_name in ['wikics']:
    dataset = WikiCS(root='../dataset/wikics')

data = dataset[0]
data = data.to(device)
num_nodes = data.num_nodes

x = data.x
x_np = x.cpu().numpy()
similarity = cosine_similarity(x_np)


edge_index = data.edge_index
adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]).to(device),
                                       torch.Size([x.shape[0], x.shape[0]])).to_dense()
x_1 = torch.torch.matmul(adj,x)
x_2 = torch.torch.matmul(adj,x_1)

adj_np = adj.cpu().numpy()
G = nx.from_numpy_matrix(adj_np)

x_list = [x,x_1,x_2]
#

Lx_update_self= [ [] for i in range(num_nodes) ]
Lx_update_sum = [ [] for i in range(num_nodes) ]
Lx_update_avg= [ [] for i in range(num_nodes) ]
Ex_update_self= [ [] for i in range(num_nodes) ]
Ex_update_avg= [ [] for i in range(num_nodes) ]
#

def att_similar_list(node, matrix):
    targrt_vector = matrix[node]
    sim_value, index = torch.topk(targrt_vector,1)
    res_list = index.cpu().numpy().tolist()
    return res_list

#
def get_maxdegree(node_list, degree_dict):
    from operator import itemgetter
    degree_list = itemgetter(*node_list)(degree_dict)
    return max(degree_list)

def set(tensor,i):
    list = tensor.tolist()
    index = list.index(i)
    list[0], list[index] = i, list[0]
    return list
#
def list_to_tensor(node_list, dim):
    max_degree = get_maxdegree(node_list, G.degree)
    ego_fea_list = []
    k_hop = 1
    for i in node_list:
        subset = k_hop_subgraph([i], k_hop, edge_index)
        #idx = set(subset[0],i)
        idx = subset[0]
        for j in range(3):
            x_i_ego_0 = x_list[j][idx, :]
            #x_i_ego_0 = x[idx, :]
            x_i_ego_0 = x_i_ego_0.resize_(max_degree+1,x_i_ego_0.shape[1])
            ego_fea_list.append(x_i_ego_0)

    res = torch.stack(ego_fea_list, dim=dim)
    trpca = TRPCA()
    res = res.cuda()
    L, E = trpca.ADMM(res)
    return L, E
#
def nlist_to_tensor(node_list, dim):
    max_degree = get_maxdegree(node_list, G.degree)
    ego_fea_list = []
    k_hop = 1
    for j in range(3):
        for i in node_list:
            subset = k_hop_subgraph([i], k_hop, edge_index)
            idx = set(subset[0],i)
            x_i_ego_0 = x_list[j][idx, :]
            x_i_ego_0 = x_i_ego_0.resize_(max_degree+1,x_i_ego_0.shape[1])
            ego_fea_list.append(x_i_ego_0)

    res = torch.stack(ego_fea_list, dim=dim)
    trpca = TRPCA()
    res = res.cuda()
    L, E = trpca.ADMM(res)
    return L, E
#
att_similar_matrix = torch.from_numpy(similarity).to(device)

dim = 0
for i in range(num_nodes): #num_nodes
    print(i)
    att_node_list = att_similar_list(i, att_similar_matrix)
    similar_node_list = att_node_list

    if i not in similar_node_list:
        similar_node_list.insert(0, i)
    index = similar_node_list.index(i)
    similar_node_list[0], similar_node_list[index] = i, similar_node_list[0]
    #print(similar_node_list)
    L_res, E_res = nlist_to_tensor(similar_node_list,dim)
    #L_res, E_res = list_to_tensor(similar_node_list,dim)  #for chameleon

    L_res = L_res.cpu().numpy()
    E_res = E_res.cpu().numpy()

    if dim == 2:
        Lx_update_self[i] = L_res[0, :, 0]
        Lx_update_avg[i] = np.sum(L_res[:, :, 0:len(similar_node_list)], axis=2, keepdims=False)[0]/len(similar_node_list)
        Lx_update_sum[i] = np.sum(L_res[:, :, 0:3], axis=0, keepdims=False)[0]
        Ex_update_self[i] = E_res[0, :, 0]
        #Ex_update_avg[i] = np.sum(E_res[:, :, 0:len(similar_node_list)], axis=2, keepdims=False)[0] / 4
    else:
        Lx_update_self[i] = L_res[0, 0, :]
        Lx_update_avg[i] = np.sum(L_res[0:len(similar_node_list), :, :], axis=0, keepdims=False)[0]/len(similar_node_list)
        Lx_update_sum[i] = np.sum(L_res[0:3, :, :], axis=0, keepdims=False)[0]
        Ex_update_self[i] = E_res[0, 0, :]
        #Ex_update_avg[i] = np.sum(E_res[0:len(similar_node_list), :, :], axis=0, keepdims=False)[0] / 4


def ego_avg(features, k):
    if(k==0):return features
    temp_fea = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        subset = k_hop_subgraph([i], k, edge_index)
        node_subset = subset[0].cpu().numpy()
        if len(node_subset) <= 1:
            temp_fea[i] = features[i]
        else:
            sum_ego = features[node_subset, :]
            temp_fea[i] = np.sum(sum_ego, axis=0, keepdims=False) / len(node_subset)
    temp_fea = np.array(temp_fea)
    return temp_fea

def pre_fea_homo(features):
    tmp = np.array(features)
    tmp = ego_avg(ego_avg(tmp,1),1)
    fea = torch.from_numpy(tmp)
    print(fea.size())
    fea = F.normalize(fea.float(), p=2, dim=1)
    return fea

def pre_fea_heter(features):
    tmp = np.array(features)
    fea = torch.from_numpy(tmp)
    print(fea.size())
    fea = F.normalize(fea.float(), p=2, dim=1)
    fea = F.normalize(fea.float(), p=2, dim=0)
    return fea

if dataset_name in homo_set:
    Lx_update_self = pre_fea_homo(Lx_update_self).cpu().numpy()
    Lx_update_avg = pre_fea_homo(Lx_update_avg).cpu().numpy()
    np.save('../my_data/' + dataset_name + '/'+dataset_name+'_tensor_feats_self.npy', Lx_update_self)
    np.save('../my_data/' + dataset_name + '/'+dataset_name+'_tensor_feats_avg.npy', Lx_update_avg)

if dataset_name in ['chameleon','squirrel']:
    Lx_update_self = pre_fea_heter(Lx_update_self).cpu().numpy()
    Lx_update_sum = pre_fea_heter(Lx_update_sum).cpu().numpy()
    np.save('../my_data/' + dataset_name + '/' + dataset_name + '_tensor_feats_avg.npy', Lx_update_sum)

if dataset_name in ['Cornell','Texas','Winconsin']:
    #Ex_update_self = pre_fea_heter(Lx_update_self).cpu().numpy()
    np.save('../my_data/' + dataset_name + '/' + dataset_name + '_tensor_feats_self.npy', Ex_update_self)


