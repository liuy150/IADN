from numpy.lib.function_base import _DIMENSION_NAME
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy
import math
from torch_scatter import scatter
import numpy as np
import torch_sparse
import scipy.sparse as sp 
from torch_geometric.nn import GCNConv as gcn

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable
        
class DisentangleGraph(nn.Module):
    def __init__(self, opt, dim, alpha, e=0.3, t=10.0):
        super(DisentangleGraph, self).__init__()
        # Disentangling Hypergraph with given H and latent_feature
        self.opt = opt
        self.latent_dim = dim   # Disentangled feature dim
        self.e = e              # sparsity parameters
        self.t = t              
        self.w = nn.Parameter(torch.Tensor(self.latent_dim, self.latent_dim))
        self.w1 = nn.Parameter(torch.Tensor(self.latent_dim, 1))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, H, int_emb):  
        """int_emb（512，1， 64）
        Input: intent-aware hidden:(Batchsize, N, dim), incidence matrix H:(batchsize, N, num_edge), intention_emb: (num_factor, dim), node mask:(batchsize, N)
        Output: Distangeled incidence matrix
        """
        node_num = hidden.shape[0]  
        select_k = self.e * node_num
        select_k = trans_to_cuda(torch.tensor([float(math.floor(select_k))])) 
        h = hidden  
        N = H.shape[0]   
        k = int_emb.shape[0]   

        select_k = select_k.repeat(1, N, k)

        int_emb =  int_emb.unsqueeze(1).repeat(1, N, 1, 1)      
        hs = h.unsqueeze(1).repeat(1, 1, k, 1)                  

        # CosineSimilarity 
        cos = nn.CosineSimilarity(dim=-1)
        sim_val = self.t * cos(hs, int_emb)                      
        
        sim_val = sim_val
        
        # sort
        _, indices = torch.sort(sim_val, dim=1, descending=True)
        _, idx = torch.sort(indices, dim=1)

        # select according to <=0
        judge_vec = idx - select_k  
        ones_vec = 2*torch.ones_like(sim_val)
        zeros_vec = torch.zeros_like(sim_val)    
        
        # intent hyperedges
        int_H = torch.where(judge_vec <= 0, ones_vec, zeros_vec)  
        int_H = int_H.squeeze(0)   # (14449,1)
        # add intent hyperedge
        if H.is_sparse:
            H = H.coalesce()
            indices = H.indices()
            values = H.values()
            int_H_indices = torch.nonzero(int_H, as_tuple=False).t()
            int_H_values = int_H[int_H_indices[0], int_H_indices[1]]
            int_H_indices[1] += H.size(1)
            new_indices = torch.cat([indices, int_H_indices], dim=1)
            new_values = torch.cat([values, int_H_values], dim=0)
            H_out = torch.sparse.FloatTensor(new_indices, new_values, (H.size(0), H.size(1) + int_H.size(1)))

        else:
            H_out = torch.cat([int_H, H], dim=-1)  # (batchsize, N, num_edge+1)

        H_out_coo = H_out.coalesce()
        row = H_out_coo.indices()[0].cpu().numpy()
        col = H_out_coo.indices()[1].cpu().numpy()
        data = H_out_coo.values().cpu().numpy()
        H_scipy = sp.coo_matrix((data, (row, col)), shape=(H_out.size(0), H_out.size(1)))

        degV = torch.from_numpy(H_scipy.sum(1)).view(-1, 1).float()   # (2708,1)  # H.sum(1)

        (row,col), values = torch_sparse.from_scipy(H_scipy)
        vertex, edges = row.cuda(), col.cuda()
        degV = degV.cuda()
        degE = scatter(degV[vertex], edges, dim=0, reduce='mean')  
        degE = degE.pow(-0.5)
        degV = degV.pow(-0.5)
        degV[degV.isinf()] = 1  
        return H_out, vertex, edges, degV, degE


    
def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X

class UniGCNIIConv(nn.Module):
    def __init__(self, opt, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.args = opt
        if self.args.journal and self.args.title:
            self.fc = nn.Linear(self.args.title_vec+self.args.journal_vec, in_features, bias=True)
        elif self.args.journal and not self.args.title:
            self.fc = nn.Linear(self.args.journal_vec, in_features, bias=True)
        elif  not self.args.journal and self.args.title:
            self.fc = nn.Linear(self.args.title_vec, in_features, bias=True)
    
    def forward(self, X, vertex, edges, degV, degE, H, alpha, beta, X0):
        vertex, edges, degV, degE = vertex, edges, degV, degE
        N = X.shape[0]
        M = H.shape[1]

      
        Xve = X[vertex] # [nnz, C]  
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate)
        Xe = Xe * degE  
        # edge -> node
        Xev = Xe[edges]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) 
        Xv = Xv * degV
        X = Xv  

        if self.args.use_norm:
            X = normalize_l2(X)

        Xi = (1-alpha) * X + alpha * X0
        X = (1-beta) * Xi + beta * self.W(Xi)

        return X
       
    
class UniGCNII(nn.Module):
    def __init__(self, args, nfeat, nhid, nlayer, nhead):
        super().__init__()
        nhid = nhid * nhead
        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(nfeat, nhid))
        for _ in range(nlayer):
            self.convs.append(UniGCNIIConv(args, nhid, nhid))
        self.convs.append(torch.nn.Linear(nhid, nfeat))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, H, vertex, edges, degV, degE):
        vertex, edges, degV, degE, H =  vertex, edges, degV, degE, H
        lamda, alpha = 0.5, 0.1 
        x = self.dropout(x)   # (n_node, dim) 14449,16
        x = F.relu(self.convs[0](x))
        x0 = x 
        for i, con in enumerate(self.convs[1:-1]):
            x = self.dropout(x)
            beta = math.log(lamda/(i+1)+1)
            # x = F.relu(con(x, V, E, Weighted, alpha, beta, x0, node_features))
            x = F.relu(con(x, vertex, edges, degV, degE, H, alpha, beta, x0))
        x = self.dropout(x)  # 14449,4
        x = self.convs[-1](x)   # (14449,16)
        return x

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = gcn(in_channels, hidden_channels)
        self.conv2 = gcn(hidden_channels, out_channels)
        self.linear = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x, adj_mat):
        x = self.conv1(x, adj_mat)
        x = F.relu(x)
        x = self.conv2(x, adj_mat)
        x = F.relu(x)
        x = self.linear(x)
        return F.log_softmax(x, dim = 1)




