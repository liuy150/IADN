import numpy as np
import torch
import scipy.sparse as sp 
from scipy.sparse import coo_matrix, csr_matrix
from collections import defaultdict
from itertools import combinations
from typing import List, Dict, Tuple
from torch.utils.data import Dataset
from tqdm import tqdm
from layers import *
import networkx as nx


def _bi_norm_lap(adj):
    # D^{-1/2}AD^{-1/2}
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return bi_lap.tocoo()

def kg_create(kg_data):

    can_triplets_np = kg_data.values
    # 获取反向三元组
    inv_triplets_np = can_triplets_np.copy()
    inv_triplets_np[:, 0] = can_triplets_np[:, 1]
    inv_triplets_np[:, 1] = can_triplets_np[:, 0]
    inv_triplets_np[:, 2] = can_triplets_np[:, 2] + np.max(can_triplets_np[:, 2]) + 1
    inv_triplets_np[:, 3] = can_triplets_np[:, 3]
    # 合并原始三元组和反向三元组
    triplets = np.vstack((can_triplets_np, inv_triplets_np))
    entities_num = max(np.max(triplets[:, 0]), np.max(triplets[:, 1])) + 1
    edge_num = np.max(triplets[:, 2]) + 1

    kg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)
    relation_set = set()
    for h_id, t_id, r_id, weight_ in tqdm(triplets, ascii=True):
        kg_graph.add_edge(h_id, t_id, key=r_id, weight=weight_)
        rd[r_id].append([h_id, t_id, weight_])
        relation_set.add(r_id)
    
    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    # for r_id in tqdm(kg_graph.keys()):
        # np_mat = np.array(kg_graph[r_id])
        # vals = np_mat[:, 2]
        # adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(enties_num, enties_num))
        # adj_mat_list.append(adj)
    for r_id in tqdm(sorted(relation_set), ascii=True):
        if rd[r_id]:
            h, t, w = zip(*rd[r_id])
            adj = sp.coo_matrix((w, (h, t)), shape=(entities_num, entities_num))
            adj_mat_list.append(adj)

    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
 

    return kg_graph, norm_mat_list, entities_num, edge_num



def inter_matrix(n_users, inter_file):
    users = inter_file['user_id'].tolist()
    items = inter_file['item_id'].tolist()
    data = [1] * len(users)
    num_users = n_users
    num_items = len(set(items))
    mat = coo_matrix(
            (data, (users, items)), shape=(num_users, num_items)
        )
    return mat


def get_edges_weight(citings, citations,node_features):
    weighteds = {}
    for citing, citation in zip(citings, citations):
        citing_text = node_features[citing]
        for cit in citation.split(','):
            cit = int(cit)
            cit_text = node_features[cit]
            sim = np.dot(citing_text, cit_text)  # 计算相似度作为权重
            weighteds[(citing, cit)] = sim

    return weighteds

def co_show_weight(citings, citations): 
    co_occurrence_count = defaultdict(int)
    for citation in citations:
        int_citations = [int(cit) for cit in citation]  
        for cit_i, cit_j in combinations(int_citations, 2):
            pair = (min(cit_i, cit_j), max(cit_i, cit_j))
            co_occurrence_count[pair] += 1 

    return dict(co_occurrence_count)

def load_graph(hypergraph_data, title_vectors, journal_vectors, num_node, opt):
    citations = hypergraph_data['cited'].tolist()
    citings = hypergraph_data['id'].tolist()
    target = hypergraph_data['target'].tolist()

    title_vectors, journal_vectors = torch.tensor(title_vectors).float(), torch.tensor(journal_vectors).float()
    if opt.journal and opt.title:
        node_features = torch.cat([title_vectors, journal_vectors], dim=1)
    elif opt.journal and not opt.title:
        node_features = journal_vectors
    elif not opt.journal and opt.title:
        node_features = title_vectors
    node_features_tensor = trans_to_cuda(node_features)
    N, M = num_node, len(citations)  # N: node number; M: edge number
    # edges_weight_dict = get_edges_weight(citings, citations,node_features
    co_occurrence_count = co_show_weight(citings, citations)
    
    indptr, indices, data, len_data = [0], [], [], [] 
    row, col, co_show, citation_weight = [], [], [], []
    for citing, cited_papers in zip(citings, citations):
        citing_text = title_vectors[citing]
        len_data.append(len(cited_papers))
        for i in range(len(cited_papers)):
            cit_text = title_vectors[cited_papers[i]]
            sim = np.dot(citing_text, cit_text)
            indices.append(cited_papers[i])
            data.append(sim)
            if opt.co_show:
                for j in range(len(cited_papers)):
                    pair = (min(cited_papers[i], cited_papers[j]), max(cited_papers[i], cited_papers[j]))
                    row.append(cited_papers[i])
                    col.append(cited_papers[j])
                    co_show.append(co_occurrence_count.get(pair,0))
            elif opt.citation:
                row.append(citing)
                col.append(cited_papers[i])
                citation_weight.append(1)

        indptr.append(len(indices))
    MAX_CIT_LEN = max(len_data)

    if opt.model == 'Intent_UniGCNII':
        H = sp.csc_matrix((data, indices, indptr), shape=(N, M), dtype=np.float32).tocsr() # V x E   # (14449,9817)
        H = normalise(H)
        H_tensor = generate_sparse_graph(H)
        if opt.gcn:
            row = np.array(row)
            col = np.array(col)
            if opt.co_show:
                data = np.array(co_show)
                H = coo_matrix((data, (row, col)), shape=(N, N), dtype=np.float32).tocsr()
            else: 
                data = np.array(citation_weight)
                H = coo_matrix((data, (row, col)), shape=(N, N), dtype=np.float32).tocsr()
                H = H + H.T
            H_content_tensor = generate_sparse_graph(H)

        else:
            H_content_tensor = None
     
    elif opt.model == 'GCN_rec':
        row = np.array(row)
        col = np.array(col)
        if opt.co_show:
            data = np.array(co_show)
        else: 
            data = np.array(citation_weight)
        H = coo_matrix((data, (row, col)), shape=(N, N), dtype=np.float32).tocsr()
        H = H + H.T
        H_tensor = generate_sparse_graph(H)
        H_content_tensor = None

    else:
        H_tensor = None
        H_content_tensor = None

    return node_features_tensor, H_tensor, H_content_tensor, MAX_CIT_LEN

def generate_sparse_graph(H):
    coo = H.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    # 转换为 PyTorch 稀疏张量
    i = torch.LongTensor(indices).cuda()
    v = torch.FloatTensor(values).cuda()
    shape = coo.shape
    H_tensor = (torch.sparse.FloatTensor(i, v, torch.Size(shape)))

    return trans_to_cuda(H_tensor)

def normalise(M):
    """
    row-normalise sparse matrix
    arguments:
    M: scipy sparse matrix
    returns:
    D^{-1} M  
    where D is the diagonal node-degree matrix 
    """
    d = np.array(M.sum(1))  # 
    
    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)    # D inverse i.e. D^{-1}
    
    return DI.dot(M)

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def map_data(Data):
    s_data = Data[0]
    s_target = Data[1]
    cur_data = []
    cur_target = []
    for i in range(len(s_data)):
        data = s_data[i]
        target = s_target[i]
        if len(data) > 40:
            continue
        cur_data.append(data)
        cur_target.append(target)
    return [cur_data, cur_target]

def handle_data(citation,MAX_CIT_LEN, opt):
    us_pois, len_data = [], []
    for cits in citation:
        le = len(cits)
        len_data.append(le)   # 记录每个session的长度
        if le < MAX_CIT_LEN:
            us_pois.append(list(reversed(cits)) + [0] * (MAX_CIT_LEN - le))
        else:
            us_pois.append(list(reversed(cits[-MAX_CIT_LEN:])))

    return us_pois, len_data


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable.cpu()
    

class Data(Dataset):
    def __init__(self, data, MAX_CIT_LEN, opt, n_node):
        self.n_node = n_node
        inputs, seq_len = handle_data(data['cited'], MAX_CIT_LEN, opt)  # 为滑动窗口准备数据，
        self.inputs = np.asarray(inputs)
        self.seq_len = np.asarray(seq_len)
        self.targets = np.asarray(data['target'])  # len: 79573
        self.citing_ids = np.asarray(data['id'])  # 79675
        self.length = len(data['cited'])
        self.max_len = MAX_CIT_LEN # max_node_num
        self.opt = opt

    def __getitem__(self, index):
        u_input, target = self.inputs[index], self.targets[index]
        citng_id = self.citing_ids[index]
        max_n_node = self.max_len

        node = np.unique(u_input)
        h_items = node.tolist() + (max_n_node - len(node)) * [0]    # padding 0

        seq_len = self.seq_len[index]

        return [torch.tensor(citng_id), torch.tensor(h_items),torch.tensor(target),torch.tensor(u_input),torch.tensor(seq_len)]

    def __len__(self):
        return self.length
