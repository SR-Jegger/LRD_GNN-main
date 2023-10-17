import networkx as nx
import numpy as np
import torch
import math

from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB, Actor, LINKXDataset, Amazon, Coauthor
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

from TRPCA_torch import TRPCA



def probability(G):
    degree_list = G.degree()
    other_degs = 0
    prob_dict = {}

    for target_node in G.nodes:
        sublist = [n for n in G.neighbors(target_node)]
        target_degree = degree_list[target_node]
        for i in sublist:
            other_degs += degree_list[i]

        if other_degs + target_degree==0:
            prob = 0
        else:
            prob = target_degree / (other_degs + target_degree)
        prob_dict[target_node] = prob
        # if target_node == 1:
        #     print(target_node,sublist,target_degree,other_degs,prob)
        other_degs = 0
    return prob_dict

def shannon(target,subnodes, prob_dict):
    shannonEnt = 0
    for node in subnodes:
        prob = prob_dict[node]
        if prob == 0: prob=1
        temp = prob * math.log(prob,10)
        shannonEnt += temp
    shannonEnt = -shannonEnt
    return shannonEnt     #Hs(subG)

def cal_Hs(G, k, node):
    G1_dict = nx.single_source_shortest_path_length(G, node, cutoff=k)
    sublist = []
    for key, v in G1_dict.items():
            sublist.append(key)
    sub_G = G.subgraph(sublist)
    res = shannon(sub_G, sublist)
    return res

def cal_Hs_k(G, prob_dict, node):
    sublist = [n for n in G.neighbors(node)]
    sublist.append(node)
    res = shannon(node,sublist, prob_dict)
    if node ==1:
       print(sublist)
    return res

def nodes2dict(G2, k, prob_dict):
    dict = {}
    if k == 1:
        for node in G2.nodes:
            #h_node = np.zeros(k)
            shannon_j = cal_Hs_k(G2, prob_dict, node)
            h_node = shannon_j
            dict[node] = h_node
    else:
        for node in G2.nodes:
            h_node = np.zeros(k)
            for j in range(1, k + 1):
                shannon_j = cal_Hs_k(G2, j, node)
                h_node[j - 1] = shannon_j
            dict[node] = h_node
    return dict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


dataset_name = 'Texas'


if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root='../dataset', name=dataset_name)
elif dataset_name in ['chameleon', 'squirrel']:
    dataset = WikipediaNetwork(root='../dataset', name=dataset_name)
elif dataset_name in ['Cornell', 'Wisconsin','Texas']:
    dataset = WebKB(root='../dataset', name=dataset_name)
elif dataset_name in ['CS', 'Physics']:
    dataset = Coauthor(root='../dataset', name=dataset_name)
elif dataset_name in ['Actor']:
    dataset = Actor(root='../dataset/Actor')
elif dataset_name in ['Photo', 'Computers']:
    dataset = Amazon(root='../dataset', name=dataset_name)
elif dataset_name in ['penn94']:
    dataset = LINKXDataset(root='./dataset', name=dataset_name)
#data = dataset.data
data = dataset[0]
data = data.to(device)
num_nodes = data.num_nodes
x = data.x
x_np = x.cpu().numpy()
similarity = cosine_similarity(x_np)

x = F.normalize(x.float(),p=1,dim=0)

edge_index = data.edge_index
adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]).to(device),
                                       torch.Size([x.shape[0], x.shape[0]])).to_dense()
adj_np = adj.cpu().numpy()
G = nx.from_numpy_matrix(adj_np)
#prob_dict = nodes2dict(G,1)
prob_dict = probability(G)
db_dict = nodes2dict(G,1,prob_dict)

sort_db_dict = sorted(db_dict.items(),  key=lambda d: d[1], reverse=False)
sort_node_list = [x for x,_ in sort_db_dict]


Lx_update_self= [ [] for i in range(num_nodes) ]
Lx_update_avg= [ [] for i in range(num_nodes) ]
Lx_update_sum= [ [] for i in range(num_nodes) ]
Ex_update_self= [ [] for i in range(num_nodes) ]

def cal_similarity(num_nodes,X):
    similar_matrix = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            print(i,j)
            similar_matrix[i,j] = torch.cosine_similarity(X[i], X[j], dim=0)
    return similar_matrix


"!!!!!"
def att_similar_list(node, matrix):
    targrt_vector = matrix[node]
    sim_value, index = torch.topk(targrt_vector,10)
    res_list = index.cpu().numpy().tolist()
    return res_list

def get_batch_list(sort_list, node):
    #res = []
    index = sort_node_list.index(node)
    if index < 21:  #21
        res = sort_list[:41] #41
    elif index > (len(sort_list)-41):
        res = sort_list[len(sort_list)-41:]
    else:
        res = sort_list[index-20:index+21]
    tmp_index = res.index(node)
    res[0],res[tmp_index] = node,res[0]
    return res


def get_maxdegree(node_list, degree_dict):
    from operator import itemgetter
    degree_list = itemgetter(*node_list)(degree_dict)
    return max(degree_list)


def list_to_tensor(node_list):
    max_degree = get_maxdegree(node_list, G.degree)
    ego_fea_list = []
    k_hop = 1
    for i in node_list:
        subset = k_hop_subgraph([i], k_hop, edge_index)
        x_i_ego_0 = x[subset[0], :]
        x_i_ego_0 = x_i_ego_0.resize_(max_degree+1,x_i_ego_0.shape[1])
        #x_i_ego_0 = x_i_ego_0.resize_(x_i_ego_0.shape[1], x_i_ego_0.shape[1])
        ego_fea_list.append(x_i_ego_0)

    res = torch.stack(ego_fea_list, dim=2)
    trpca = TRPCA()
    #res = res.cpu().numpy()#.cuda()
    res = res.cuda()
    #L, E = trpca.ADMM(res)
    L,E = trpca.T_SVD(res,1)

    return L,E


#att_similar_matrix = torch.from_numpy(similarity).to(device)


