import torch
import torch_sparse
from ogb.graphproppred.mol_encoder import BondEncoder, AtomEncoder
from torch import nn as nn
from torch.nn import PReLU, Linear, Sequential, Tanh, ReLU, ELU, BatchNorm1d as BN
from torch.nn import functional as F
from torch_geometric import nn as nng
from torch_geometric.data import Batch, Data
from torch_sparse import SparseTensor, coalesce
from torch_scatter import scatter_add
from copy import copy
# from utils.attention import MultiheadAttention

import math
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch import Tensor
import torch_geometric
from torch_geometric.utils import degree as node_degree

class APPNP(nn.Module):
    def __init__(self, K=5, alpha=0.8):
        super().__init__()
        self.appnp = nng.conv.APPNP(K=K, alpha=alpha)
    
    def forward(self, data):
        data = new(data)
        x, ei, ea, b = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.appnp(x, ei)
        data.x = h
        return data

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.main = [
            nn.Linear(in_dim, 2 * in_dim),
            nn.BatchNorm1d(2 * in_dim),
            nn.ReLU()
        ]
        self.main.append(nn.Linear(2 * in_dim, out_dim))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)


#################
# Embeddings
#################
import numpy, scipy.sparse
import scipy.sparse as sp
import numpy as np

def get_adj_matrix(edge_index_fea, N):
    adj = torch.zeros([N, N])
    adj[edge_index_fea[0, :], edge_index_fea[1, :]] = 1
    Asp = scipy.sparse.csr_matrix(adj)
    Asp = Asp + Asp.T.multiply(Asp.T > Asp) - Asp.multiply(Asp.T > Asp)
    Asp = Asp + sp.eye(Asp.shape[0])

    D1_ = np.array(Asp.sum(axis=1))**(-0.5)
    D2_ = np.array(Asp.sum(axis=0))**(-0.5)
    D1_ = sp.diags(D1_[:,0], format='csr')
    D2_ = sp.diags(D2_[0,:], format='csr')
    A_ = Asp.dot(D1_)
    A_ = D2_.dot(A_)
    A_ = sparse_mx_to_torch_sparse_tensor(A_)
    return A_

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def propagate(feature, A, order):
    x = feature
    y = feature
    for i in range(order):
        x = torch.spmm(A, x).detach_()
        y.add_(x)
    return y.div_(order+1.0).detach_()

class OGBMolEmbedding(nn.Module):
    def __init__(self, dim, embed_edge=True, x_as_list=False, degree=False):
        super().__init__()
        self.atom_embedding = AtomEncoder(emb_dim=dim)
        self.degree = degree
        if embed_edge:
            self.edge_embedding = BondEncoder(emb_dim=dim)
        if degree:
            self.degree_emb = nn.Embedding(64, dim)
        self.x_as_list = x_as_list

    def forward(self,data):
        data = new(data)
        # print("epoch recieved here")
        start = end = 0
#         if data.epoch > data.reg_epochs:
#             new_fea_list = []
#             for i in range(data.y.shape[0]):
#                 new_fea = self.atom_embedding(data.x[data.batch == i])
#                 end = start + (data.batch == i).sum().item()
#                 edge_index_fea = data.edge_index[:,(data.edge_index[0] >= start) * (data.edge_index[0] < end)] - start
#                 # print(start,end,data.batch.shape,len(data.batch),data.batch,len(data.edge_index[0]),edge_index_fea)
#                 start = end
#                 N = new_fea.size(0)
#                 drop_rate=0.2 

#                 if self.training:
#                     drop_rates = torch.FloatTensor(np.ones(N) * drop_rate)
#                     masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
#                     new_fea = masks.cuda() * new_fea
#                 else:
#                     new_fea = new_fea*(1.-drop_rate)
#                 ori_fea = new_fea
#                 adj = get_adj_matrix(edge_index_fea, N).to(data.edge_index.device)
#                 # print('successful')
#                 order = 1
#                 new_fea = propagate(new_fea, adj, order)
#                 new_fea_list.append(new_fea)
#             data.x = torch.cat(new_fea_list, dim =0)
#             data.x = data.x + self.degree_emb(data.in_degree) if self.degree else data.x
            
#         else:
        data.x = self.atom_embedding(data.x) + self.degree_emb(data.in_degree) if self.degree else self.atom_embedding(data.x)
    
    
        if data.perturb is not None:
            data.x = data.x + data.perturb
        if self.x_as_list:
            data.x = [data.x]
        if hasattr(self, 'edge_embedding'):
            data.edge_attr = self.edge_embedding(data.edge_attr)
        return data


class NodeEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, x_as_list=False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=in_dim, embedding_dim=out_dim)
        self.x_as_list = x_as_list

    def forward(self, data):
        data = new(data)
        data.x = self.embedding(data.x)
        if self.x_as_list:
            data.x = [data.x]
        return data


class VNAgg(nn.Module):
    def __init__(self, dim, conv_type="gin"):
        super().__init__()
        self.conv_type = conv_type
        if "gin" in conv_type:
            self.mlp = nn.Sequential(
                MLP(dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU()
            )
        elif "gcn" in conv_type:
            self.W0 = nn.Linear(dim, dim)
            self.W1 = nn.Linear(dim, dim)
            self.nl_bn = nn.Sequential(
                nn.BatchNorm1d(dim),
                nn.ReLU()
            )
        else:
            raise NotImplementedError('Unrecognised model conv : {}'.format(conv_type))

    def forward(self, virtual_node, embeddings, batch_vector):
        if batch_vector.size(0) > 0:  # ...or the operation will crash for empty graphs
            G = nng.global_add_pool(embeddings, batch_vector)
        else:
            G = torch.zeros_like(virtual_node)
        if "gin" in self.conv_type:
            virtual_node = virtual_node + G
            virtual_node = self.mlp(virtual_node)
        elif "gcn" in self.conv_type:
            virtual_node = self.W0(virtual_node) + self.W1(G)
            virtual_node = self.nl_bn(virtual_node)
        else:
            raise NotImplementedError('Unrecognised model conv : {}'.format(self.conv_type))
        return virtual_node


class ConvBlock(nn.Module):
    def __init__(self, dim, nhead = 1, dropout=0.5, activation=F.relu, virtual_node=False, virtual_node_agg=True,
                 k=4, last_layer=False, conv_type='gin', edge_embedding=None):
        super().__init__()
        self.edge_embed = edge_embedding
        self.conv_type = conv_type
        if conv_type == 'gin+':
            self.conv = GINEPLUS(MLP(dim, dim), dim, k=k)
        elif conv_type == 'gin++':
            # self.conv = GINEATTENTION(MLP(dim, dim), dim, k=k)
            self.conv = CausalSelfAttention(MLP(dim, dim), dim, k=k, nhead=nhead)
        elif conv_type == 'naivegin+':
            self.conv = NAIVEGINEPLUS(MLP(dim, dim), dim, k=k)
        elif conv_type == 'gin':
            self.conv = nng.GINEConv(MLP(dim, dim), train_eps=True)
        elif conv_type == 'gcn':
            self.conv = nng.GCNConv(dim, dim)
        self.norm = nn.BatchNorm1d(dim)
        self.act = activation or nn.Identity()
        self.last_layer = last_layer

        self.dropout_ratio = dropout

        self.virtual_node = virtual_node
        self.virtual_node_agg = virtual_node_agg
        if self.virtual_node and self.virtual_node_agg:
            self.vn_aggregator = VNAgg(dim, conv_type=conv_type)

    def forward(self, data):
        data = new(data)
        x, ei, ea, b = data.x, data.edge_index, data.edge_attr, data.batch
        mhei, d = data.multihop_edge_index, data.distance
        h = x
        z = h[0]
        if self.virtual_node:
            if self.conv_type == 'gin+' or self.conv_type == 'gin++':
                h[0] = h[0] + data.virtual_node[b]
            else:
                h = h + data.virtual_node[b]
        if self.conv_type == 'gin':
            H = self.conv(h, ei, edge_attr=self.edge_embed(ea))
        elif self.conv_type == 'gcn':
            H = self.conv(h, ei)
        else:
            H = self.conv(h, mhei, d, self.edge_embed(ea))
        if self.conv_type == 'gin+' or self.conv_type == 'gin++':
            h = H[0]
        else:
            h = H
        h = self.norm(h)
        if not self.last_layer:
            h = self.act(h)
        h = F.dropout(h, self.dropout_ratio, training=self.training)

        if self.virtual_node and self.virtual_node_agg:
            v = self.vn_aggregator(data.virtual_node, h, b)
            v = F.dropout(v, self.dropout_ratio, training=self.training)
            data.virtual_node = v
        if self.conv_type == 'gin+' or self.conv_type == 'gin++':
            H[0] = h + z
            h = H
        data.x = h
        return data

class GlobalPool(nn.Module):
    def __init__(self, fun, cat_size=False, cat_candidates=False, hidden=0):
        super().__init__()
        self.cat_size = cat_size
        if fun in ['mean', 'max', 'add']:
            self.fun = getattr(nng, "global_{}_pool".format(fun.lower()))
        else:
            self.fun = nng.GlobalAttention(gate_nn = 
                    torch.nn.Sequential(
                        torch.nn.Linear(hidden, hidden)
                        ,torch.nn.BatchNorm1d(hidden)
                        ,torch.nn.ReLU()
                        ,torch.nn.Linear(hidden, 1)))

        self.cat_candidates = cat_candidates

    def forward(self, batch):
        x, b = batch.x, batch.batch
        pooled = self.fun(x, b, size=batch.num_graphs)
        if self.cat_size:
            sizes = nng.global_add_pool(torch.ones(x.size(0), 1).type_as(x), b, size=batch.num_graphs)
            pooled = torch.cat([pooled, sizes], dim=1)
        if self.cat_candidates:
            ei = batch.edge_index
            mask = batch.edge_attr == 3
            candidates = scatter_add(x[ei[0, mask]], b[ei[0, mask]], dim=0, dim_size=batch.num_graphs)
            pooled = torch.cat([pooled, candidates], dim=1)
        return pooled


def make_degree(data):
    data = new(data)
    N = data.num_nodes
    adj = torch.zeros([N,N], dtype=torch.bool, device=data.x.device)
    edge_index = data.edge_index
    adj[edge_index[0,:], edge_index[1,:]] = True
    data.in_degree = adj.long().sum(dim=1).view(-1)
    data.out_degree = adj.long().sum(dim=0).view(-1)
    return data

def make_multihop_edges(data, k):
    """
    Adds edges corresponding to distances up to k to a data object.
    :param data: torch_geometric.data object, in coo format
    (ie an edge (i, j) with label v is stored with an arbitrary index u as:
     edge_index[0, u] = i, edge_index[1, u]=j, edge_attr[u]=v)
    :return: a new data object with new fields, multihop_edge_index and distance.
    distance[u] contains values from 1 to k corresponding to the distance between
    multihop_edge_index[0, u] and multihop_edge_index[1, u]
    """
    data = new(data)

    N = data.num_nodes
    E = data.num_edges
    if E == 0:
        data.multihop_edge_index = torch.empty_like(data.edge_index)
        data.distance = torch.empty_like(data.multihop_edge_index[0])
        return data

    # Get the distance 0
    multihop_edge_index = torch.arange(0, N, dtype=data.edge_index[0].dtype, device=data.x.device)
    distance = torch.zeros_like(multihop_edge_index)
    multihop_edge_index = multihop_edge_index.unsqueeze(0).repeat(2, 1)

    # Get the distance 1
    multihop_edge_index = torch.cat((multihop_edge_index, data.edge_index), dim=1)
    distance = torch.cat((distance, torch.ones_like(data.edge_index[0])), dim=0)

    A = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=torch.ones_like(data.edge_index[0]).float(),
                     sparse_sizes=(N, N), is_sorted=False)
    Ad = A  # A to the power d

    # Get larger distances
    for d in range(2, k + 1):
        Ad = torch_sparse.matmul(Ad, A)
        row, col, v = Ad.coo()
        d_edge_index = torch.stack((row, col))
        d_edge_attr = torch.empty_like(row).fill_(d)
        multihop_edge_index = torch.cat((multihop_edge_index, d_edge_index), dim=1)
        distance = torch.cat((distance, d_edge_attr), dim=0)

    # remove dupicate, keep only shortest distance
    multihop_edge_index, distance = coalesce(multihop_edge_index, distance, N, N, op='min')

    data.multihop_edge_index = multihop_edge_index
    data.distance = distance

    return data


class NAIVEGINEPLUS(nng.MessagePassing):
    def __init__(self, fun, dim, k=4, **kwargs):
        super().__init__(aggr='add')
        self.k = k
        self.nn = fun
        self.eps = nn.Parameter(torch.zeros(k + 1, dim), requires_grad=True)

    def forward(self, x, multihop_edge_index, distance, edge_attr):
        assert x.size(-1) == edge_attr.size(-1)
        result = (1 + self.eps[0]) * x
        for i in range(self.k):
            if i == 0:
                out = self.propagate(multihop_edge_index[:, distance == i + 1], edge_attr=edge_attr, x=x)
            else:
                out = self.propagate(multihop_edge_index[:, distance == i + 1], edge_attr=None, x=x)
            result += (1 + self.eps[i + 1]) * out
        result = self.nn(result)
        return result

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            return F.relu(x_j + edge_attr)
        else:
            return F.relu(x_j)

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn, self.eps.size(0))


class GINEPLUS(nng.MessagePassing):
    def __init__(self, fun, dim, k=4, **kwargs):
        super().__init__(aggr='add')
        self.k = k
        self.nn = fun
        self.eps = nn.Parameter(torch.zeros(k + 1, dim), requires_grad=True)

    def forward(self, XX, multihop_edge_index, distance, edge_attr):
        """Warning, XX is now a list of previous xs, with x[0] being the last layer"""
        assert len(XX) >= self.k
        assert XX[-1].size(-1) == edge_attr.size(-1)
        result = (1 + self.eps[0]) * XX[0]
        for i, x in enumerate(XX):
            if i >= self.k:
                break
            if i == 0:
                out = self.propagate(multihop_edge_index[:, distance == i + 1], edge_attr=edge_attr, x=x)
            else:
                out = self.propagate(multihop_edge_index[:, distance == i + 1], edge_attr=None, x=x)
            result += (1 + self.eps[i + 1]) * out
        result = self.nn(result)
        return [result] + XX

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            return F.relu(x_j + edge_attr)
        else:
            return F.relu(x_j)


class GINEATTENTION(nng.MessagePassing):
    def __init__(self, fun, dim, k=4, **kwargs):
        super().__init__(aggr='add')
        self.k = k
        self.nn = fun
        self.eps = nn.Parameter(torch.zeros(k + 1, dim), requires_grad=True)
        self.attention_e = MultiheadAttention(dim, 8, encode_dim=64)

    def forward(self, XX, multihop_edge_index, distance, edge_attr):
        """Warning, XX is now a list of previous xs, with x[0] being the last layer"""
        assert len(XX) >= self.k
        assert XX[-1].size(-1) == edge_attr.size(-1)
        result = (1 + self.eps[0]) * XX[0]
        for i, x in enumerate(XX):
            if i >= self.k:
                break
            if i == 0:
                out = self.propagate(multihop_edge_index[:, distance == i + 1], edge_attr=edge_attr, x=x)
            else:
                out = self.propagate(multihop_edge_index[:, distance == i + 1], edge_attr=None, x=x)
            result += (1 + self.eps[i + 1]) * out
        result = self.nn(result)
        return [result] + XX

    def message(self, x_i, x_j, edge_attr):
        if edge_attr is not None:
            all_feature = torch.cat((x_i.unsqueeze(0), x_j.unsqueeze(0), edge_attr.unsqueeze(0)))
            out = self.attention_e(all_feature)
            return F.relu(torch.sum(out[0][1:],dim=0))
        else:
            return F.relu(x_j)

def new(data):
    """returns a new torch_geometric.data.Data containing the same tensors.
    Faster than data.clone() (tensors are not cloned) and preserves the old data object as long as
    tensors are not modified in-place. Intended to modify data object, without modyfing the tensors.
    ex:
    d1 = data(x=torch.eye(3))
    d2 = new(d1)
    d2.x = 2*d2.x
    In that case, d1.x was not changed."""
    return copy(data)



class CausalSelfAttention(nng.MessagePassing):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, fun, dim, k, nhead=8, attn_pdrop=0.1, resid_pdrop=0.1, add_self_loop = True, norm_w = 1.0, args=None):
        # super().__init__()
        super(CausalSelfAttention, self).__init__(aggr="add")
        n_embd = dim
        assert n_embd % nhead == 0
        # key, query, value projections for all heads
        self.k = k
        self.nn = fun
        self.eps = nn.Parameter(torch.zeros(k + 1, dim), requires_grad=True)
        self.fea_mlp = Sequential(
            Linear(n_embd, n_embd),
            PReLU(),
            Linear(n_embd, n_embd)
            ,PReLU())
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.norm_w = norm_w
        self.n_head = nhead

        self.mix = True
        self.add_self_loop = add_self_loop
        # self.edg_enc = BondEncoder(emb_dim=n_embd)##change fr ppa
        # self.edg_enc = nn.Linear(7, n_embd)
        # self.edge_mlp = nn.Sequential(
        #     nn.Linear(n_embd, n_embd),
        #     nn.GELU(),
        #     nn.Linear(n_embd, n_embd)
        # )
        
#     def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
#                 size_i: Optional[int],weight=None) -> Tensor:
#         # Given edge-level attention coefficients for source and target nodes,
#         # we simply need to sum them up to "emulate" concatenation:
#         B,hs = x_i.size()
#         if edge_attr is not None:
#             e_b ,e_hs = edge_attr.size()
#         q = x_i.reshape((B,3,hs//3))[:,1,:].reshape((B,self.n_head,hs//3//self.n_head))
#         k = x_j.reshape((B,3,hs//3))[:,0,:].reshape((B,self.n_head,hs//3//self.n_head))
#         v = x_j.reshape((B,3,hs//3))[:,2,:].reshape((B,self.n_head,hs//3//self.n_head))
#         if edge_attr is not None:
#             edge_attr = edge_attr.reshape((e_b,self.n_head,e_hs//self.n_head))
        
#         att = (((q * torch.ones_like(q)) * (k * torch.ones_like(k)))* (1.0 / math.sqrt(k.size(-1)))).sum(dim=-1)
#         # alpha = F.leaky_relu(alpha, self.negative_slope) this and adding edge info here both can be considered.
#         # if weight is not None:
#         #     att = softmax(att, index, ptr, size_i) * weight.reshape((-1,1))
#         # else:
#         #     att = softmax(att, index, ptr, size_i)
#         att = F.sigmoid(att)
#         att = self.attn_drop(att) ## remove later 
        
#         if edge_attr is not None:
#             v += edge_attr
        
#         if self.mix:
#             v = F.relu(v)
#         out = v * att.unsqueeze(-1)
#         out=v
#         return self.fea_mlp(out.reshape(B,-1))
#         # return out.reshape(B,-1)
    
    # def forward(self, XX, multihop_edge_index, distance, edge_attr):
    #     """Warning, XX is now a list of previous xs, with x[0] being the last layer"""
    #     assert len(XX) >= self.k
    #     assert XX[-1].size(-1) == edge_attr.size(-1)
    #     # result = (1 + self.eps[0]) * XX[0]
    #     result = (1 + self.eps[0]) * self.fea_mlp(XX[0])
    #     for i, x in enumerate(XX):
    #         if i >= self.k:
    #             break
    #         B, hs = x.size()
    #         k = self.key(x)  # (B, nh, T, hs)
    #         q = self.query(x)  # (B, nh, T, hs)
    #         v = self.value(x)  # (B, nh, T, hs)
    #         # edge_attr = self.edge_mlp(edge_attr) ##change for ppa 
    #         x = torch.cat([k,q,v],dim = -1)
    #         deg = node_degree(multihop_edge_index[1, distance == i + 1], x.size(0), dtype=x.dtype)
    #         deg_sqrt = deg.pow(0.5)
    #         weight = deg_sqrt[multihop_edge_index[1, distance == i + 1]]
    #         x = (x,x)
    #         if i == 0:
    #             out = self.propagate(multihop_edge_index[:, distance == i + 1], edge_attr=edge_attr, x=x,size=None,weight = weight)
    #         else:
    #             out = self.propagate(multihop_edge_index[:, distance == i + 1], edge_attr=None, x=x,size=None,weight = None)
    #         out = out.reshape(B,hs)
    #         out = self.resid_drop(self.proj(out))
    #         result += (1 + self.eps[i + 1]) * out
    #     result = self.nn(result) + result
    #     # result = self.nn(result)
    #     return [result] + XX
    
    def forward(self, XX, multihop_edge_index, distance, edge_attr):
        """Warning, XX is now a list of previous xs, with x[0] being the last layer"""
        assert len(XX) >= self.k
        assert XX[-1].size(-1) == edge_attr.size(-1)
        result = (1 + self.eps[0]) * XX[0]
        # result = (1 + self.eps[0]) * self.fea_mlp(XX[0])
        x = XX[0]
        i = self.k-1
        B, hs = x.size()
        k = self.key(x)  # (B, nh, T, hs)
        q = self.query(x)  # (B, nh, T, hs)
        v = self.value(x)  # (B, nh, T, hs)
        # edge_attr = self.edge_mlp(edge_attr) ##change for ppa 
        x = torch.cat([k,q,v],dim = -1)
        # deg = node_degree(multihop_edge_index[1, distance == i + 1], x.size(0), dtype=x.dtype)
        # deg_sqrt = deg.pow(0.5)
        # weight = deg_sqrt[multihop_edge_index[1, distance == i + 1]]
        x = (x,x)

        out = self.propagate(multihop_edge_index[:, (distance <= i + 1) *(distance >= 1)], edge_attr=None, x=x,size=None,weight = None)
        out = out.reshape(B,hs)
        out = self.resid_drop(self.proj(out))
        result += (1 + self.eps[i + 1]) * out
        result = self.nn(result) + result
        # result = result 
        # result = self.nn(result)
        return [result] + XX
    
    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int],weight=None) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        B,hs = x_i.size()
        if edge_attr is not None:
            e_b ,e_hs = edge_attr.size()
        q = x_i.reshape((B,3,hs//3))[:,1,:].reshape((B,self.n_head,hs//3//self.n_head))
        k = x_j.reshape((B,3,hs//3))[:,0,:].reshape((B,self.n_head,hs//3//self.n_head))
        v = x_j.reshape((B,3,hs//3))[:,2,:].reshape((B,self.n_head,hs//3//self.n_head))
        if edge_attr is not None:
            edge_attr = edge_attr.reshape((e_b,self.n_head,e_hs//self.n_head))
        
        att = (((q * torch.ones_like(q)) * (k * torch.ones_like(k)))* (1.0 / math.sqrt(k.size(-1)))).sum(dim=-1)
        # alpha = F.leaky_relu(alpha, self.negative_slope) this and adding edge info here both can be considered.
        # if weight is not None:
        #     att = softmax(att, index, ptr, size_i) * weight.reshape((-1,1))
        # else:
        #     att = softmax(att, index, ptr, size_i)
        att = F.sigmoid(att)
        att = self.attn_drop(att) ## remove later 
        
        if edge_attr is not None:
            v += edge_attr
        
        v = F.relu(v)
        out = v * att.unsqueeze(-1)
        out=v
        return self.fea_mlp(out.reshape(B,-1))
        # return out.reshape(B,-1)