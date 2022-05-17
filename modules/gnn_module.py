import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    # GlobalAttention,
    MessagePassing,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.inits import uniform
from torch_scatter import scatter
from modules.conv import GCNConv, GINConv, GATConv, GSANConv, GSANEAConv, GSANENAConv
# from modules.gatconv import GATConv
from modules.utils import pad_batch, GlobalAttention, GraphMultisetTransformer
from data.floyd_warshall import floyd_warshall_fastest, floyd_warshall_fastest_torch
from torch_scatter import scatter_add
import math
from torch_geometric.utils import  remove_self_loops, softmax

import numpy as np
import numpy, scipy.sparse
import scipy.sparse as sp


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
### GNN to generate nodse embedding
def global_mask_add_pool(x, batch, is_att, size=None):
    size = int(batch.max().item() + 1) if size is None else size
    h = x*(1-is_att.long())
    return scatter(h, batch, dim=0, dim_size=size, reduce='add')
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    @staticmethod
    def need_deg():
        return False

    def __init__(self, num_layer, emb_dim, node_encoder, edge_encoder_cls, drop_ratio=0.5, JK="last", residual=False, gnn_type="gin", gat_heads = 4, expanded = False, virtual_attention = False):
        """
        emb_dim (int): node embedding dimensionality
        num_layer (int): number of GNN message passing layers
        """

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.virtual_attention = virtual_attention
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = node_encoder

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.rates = [7.0,4.0,1.0]
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.convs.append(GINConv(emb_dim, edge_encoder_cls))
            elif gnn_type == "gcn":
                self.convs.append(GCNConv(emb_dim, edge_encoder_cls))
            elif gnn_type == "gat":
                self.convs.append(GATConv(emb_dim,emb_dim//gat_heads, gat_heads,expanded = expanded,rate=self.rates[i]))
            else:
                ValueError("Undefined GNN type called {}".format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data, perturb=None):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        edge_index_new = batched_data.edge_index_new if hasattr(batched_data, "edge_index_new") else None
        N = batched_data.num_nodes
        weight = batched_data.weight if hasattr(batched_data, "weight") else None
        node_depth = batched_data.node_depth if hasattr(batched_data, "node_depth") else None
        
        edge_index_new = []
        for i in range(len(N)):
            n = torch.sqrt(N[i]).to(torch.int32)
            row_index = torch.arange(n,device = x.device).reshape((-1,1)).repeat((1,n-1)).reshape((-1,))
            col_index = torch.arange(n, device = x.device).repeat((n,)).reshape((n,n))
            col_index[torch.eye(n, device = x.device).bool()] = -1
            col_index = col_index[col_index!=-1]
            edge_index_new.append(torch.stack((row_index,col_index)))
        edge_index_new = torch.cat(edge_index_new,dim=-1)
        N_shift = torch.roll(N,1)
        N_shift[0] = 0
        shift = torch.cat([N_shift[i].repeat(N[i].item()) for i in range(len(N))])
        edge_index_new = edge_index_new + shift
        
        torch.set_printoptions(profile="full")
        print(x.shape)
        print(weight.shape)
        # print(N)
        print(edge_index_new.shape)
#         print(edge_index)
        
#         print(edge_attr)
        exit()
        
        ### computing input node embedding
       
        if self.node_encoder is not None:
            encoded_node = (
                self.node_encoder(x)
                if node_depth is None
                else self.node_encoder(
                    x,
                    node_depth.view(
                        -1,
                    ),
                )
            )
        else:
            encoded_node = x
        tmp = encoded_node + perturb if perturb is not None else encoded_node
        h_list = [tmp]

        for layer in range(self.num_layer):

            # h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.convs[layer](h_list[layer], edge_index, edge_attr, edge_index_new, weight)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]
        elif self.JK == "cat":
            node_representation = torch.cat([h_list[0], h_list[-1]], dim=-1)

        return node_representation


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """

    @staticmethod
    def need_deg():
        return False

    def __init__(self, args, num_layer, emb_dim, node_encoder, edge_encoder_cls, drop_ratio=0.5, JK="last", residual=False, gnn_type="gin", gat_heads = 4, expanded=False, virtual_attention = False):
        """
        emb_dim (int): node embedding dimensionality
        """

        super(GNN_node_Virtualnode, self).__init__()
        self.att_node = args.gnn_att_node
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.expanded=expanded
        self.virtual_attention = virtual_attention
        self.gnn_type = gnn_type
        self.hig = args.hig
        self.hig_started = False
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = node_encoder
        # self.beta = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor([1])) for _ in range(num_layer)])
        # self.alpha = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor([1])) for _ in range(num_layer)])
        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        
        self.vn_q = torch.nn.ModuleList([torch.nn.Linear(emb_dim,emb_dim) for _ in range(num_layer)])
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        self.global_attention = GlobalAttention(emb_dim,1,num_layer) 
        self.gmt = torch.nn.ModuleList([GraphMultisetTransformer(emb_dim,emb_dim,emb_dim,num_nodes=28) for _ in range(num_layer)])
        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        
        # self.rates = [10.0,5.0,4.0,4.0,1.0,0.1]
        self.rates = [20.0,20.0,20.0,20.0]
        # [5.0,4.0,1.0,0.1]
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.convs.append(GINConv(emb_dim, edge_encoder_cls))
            elif gnn_type == "gcn":
                self.convs.append(GCNConv(emb_dim, edge_encoder_cls))
            elif gnn_type == "gat":
                self.convs.append(GATConv(emb_dim,emb_dim//gat_heads, gat_heads,edge_encoder_cls=edge_encoder_cls,fill_value=0.0))
            elif gnn_type == "gsan":
                self.convs.append(GSANConv(emb_dim, edge_encoder_cls,args))
            elif gnn_type == "gsanea":
                self.convs.append(GSANEAConv(emb_dim, edge_encoder_cls,args))
            elif gnn_type == "gsanena":
                self.convs.append(GSANENAConv(emb_dim, edge_encoder_cls,args))
            else:
                ValueError("Undefined GNN type called {}".format(gnn_type))
            
            norm_func = lambda emb_dim: torch.nn.BatchNorm1d(emb_dim)
            # norm_func = lambda emb_dim: torch.nn.Identity()
            # norm_func = lambda emb_dim: torch.nn.LayerNorm(emb_dim)
            self.batch_norms.append(norm_func(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2 * emb_dim),
                    norm_func(2 * emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * emb_dim, emb_dim),
                    norm_func(emb_dim),
                    torch.nn.ReLU(),
                )
            )
        self.edge_encoder = edge_encoder_cls(emb_dim)
    def mr_warshall(self,N,edges):
        edge_index_transposed = edges.T
        A = torch.eye(N, device = edges.device)
        C = torch.eye(N, device = edges.device)

        A[edge_index_transposed[:,1],edge_index_transposed[:,0]] = 1
        A[A==0] = float('inf')
        A = A - C
        for k in range(len(A)):
            A1 = A[k,:].unsqueeze(0)
            A2 = A.unsqueeze(2)[:,k]
            # A = torch.min(A, A[k,:].unsqueeze(0) + A.unsqueeze(2)[:,k])
            A = torch.min(A, A1+ A2)
            del A1
            del A2
        A[A==0] = 1
        A = -(A-1)
        A[A<-120] = -120.0
        # weight = A.to(torch.int8)
        A[torch.eye(N).bool()] = 1
        A = A[A!=1]
        return A
    def forward(self, batched_data, perturb=None):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        edge_index_new = batched_data.edge_index_new if hasattr(batched_data, "edge_index_new") else None
        # N = batched_data.N
        # E = batched_data.E
        weight = batched_data.weight if hasattr(batched_data, "weight") else None
        node_depth = batched_data.node_depth if hasattr(batched_data, "node_depth") else None
    
        if self.expanded:
            edge_index_new = []
            for i in range(len(N)):
                n = torch.sqrt(N[i]).to(torch.int32)
                row_index = torch.arange(n,device = x.device).reshape((-1,1)).repeat((1,n-1)).reshape((-1,))
                col_index = torch.arange(n, device = x.device).repeat((n,)).reshape((n,n))
                col_index[torch.eye(n, device = x.device).bool()] = -1
                col_index = col_index[col_index!=-1]
                edge_index_new.append(torch.stack((row_index,col_index)))
            edge_index_new = torch.cat(edge_index_new,dim=-1)

            N_root = torch.sqrt(N).to(torch.int32)
            N_shift = torch.roll(N_root,1)
            N_shift[0] = 0
            N_shift = torch.cumsum(N_shift,dim=0)
            N = (N - torch.sqrt(N)).to(torch.int)
            shift = torch.cat([N_shift[i].repeat(N[i].item()) for i in range(len(N))])
            edge_index_new = edge_index_new + shift

            E = torch.cumsum(E,dim=0)
            E = F.pad(E,(1,0))
            weight = []
            for i in range(len(E)-1):
                e_i = edge_index[:,E[i]:E[i+1]]
                e_i_corrected = e_i - N_shift[i]
                w_i = self.mr_warshall(N_root[i], e_i_corrected)

                weight.append(w_i)

            weight = torch.cat(weight)    
        
        # print(weight.shape)
        # print(edge_index_new.shape)
        # print(batched_data.num_nodes)
        # torch.set_printoptions(profile="full")
        # print(x.shape)
        # print(weight.shape)
        # print(N.shape)
        # print(edge_index_new.shape, edge_index_new)
        # print(edge_index)
        # exit()
        
        ### computing input node embedding
        # print("value of x",x)
        
        if self.node_encoder is not None:
            encoded_node = (self.node_encoder(x) if node_depth is None else self.node_encoder(x,node_depth.view(-1,),))
#             # print(type(batched_data.adj),len(batched_data.adj))
#             # exit()
#             #handling hig
#             start = end = 0
#             if self.hig and self.hig_started:
#                 new_fea_list = []
#                 for i in range(batched_data.y.shape[0]):
#                     new_fea = self.node_encoder(x[batch == i])
#                     # end = start + (batch == i).sum().item()
#                     # edge_index_fea = edge_index[:,(edge_index[0] >= start) * (edge_index[0] < end)] - start
#                     # # print(start,end,data.batch.shape,len(data.batch),data.batch,len(data.edge_index[0]),edge_index_fea)
#                     # start = end
#                     N = new_fea.size(0)
#                     drop_rate=0.2 

#                     if self.training:
#                         drop_rates = torch.FloatTensor(np.ones(N) * drop_rate)
#                         masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
#                         new_fea = masks.cuda() * new_fea
#                     else:
#                         new_fea = new_fea*(1.-drop_rate)
#                     ori_fea = new_fea
#                     # adj = get_adj_matrix(edge_index_fea, N).to(edge_index.device)
#                     adj = sparse_mx_to_torch_sparse_tensor(batched_data.adj[i]).to(edge_index.device)
#                     # print('successful')
#                     order = 1
#                     new_fea = propagate(new_fea, adj, order)
#                     new_fea_list.append(new_fea)
#                 encoded_node = torch.cat(new_fea_list, dim =0)
        else:
            encoded_node = x
        # encoded_node = x
        tmp = encoded_node + perturb if perturb is not None else encoded_node
        h_list = [tmp]
        ### virtual node embeddings for graphs
        # print(len(batch), batch[-1])
        # print(self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1)))
        # exit()
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            if self.virtual_attention and layer > 0:
#                 q = self.vn_q[layer-1](h_list[layer])
#                 k = self.vn_k[layer-1](virtualnode_embedding)[batch]
#                 v = self.vn_v[layer-1](virtualnode_embedding)[batch]

#                 att = (((q * torch.ones_like(q)) * (k * torch.ones_like(k)))* (1.0 / math.sqrt(k.size(-1)))).sum(dim=-1)
#                 att = F.tanh(att).unsqueeze(-1)
#                 v = att*v
#                 h_list[layer] = h_list[layer] + v
                h_list[layer] = h_list[layer] + virtualnode_embedding[batch]
            else:
                if not self.att_node:
                    h_list[layer] = h_list[layer] + virtualnode_embedding[batch]
                # h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            if self.gnn_type in ['gsanea','gsanena']:
                h, edge_index, edge_attr = self.convs[layer](h_list[layer], edge_index, edge_attr)
            else:
                h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            # h = self.convs[layer](h_list[layer], edge_index, edge_attr, edge_index_new, weight)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            
            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                if self.virtual_attention:
                    
                    #for replacing the position of virtual node mlp
                    # if layer!=0:
                    #     virtualnode_embedding = virtualnode_embedding_temp2 + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),self.drop_ratio, training=self.training)

#                     q = self.vn_q[layer](virtualnode_embedding)[batch]
#                     # k = self.vn_k[layer](h_list[layer])
#                     # v = self.vn_v[layer](h_list[layer])
#                     k,v = h_list[layer], h_list[layer]
#                     q_vn = self.vn_q[layer](virtualnode_embedding)
#                     # k_vn = self.vn_k[layer](virtualnode_embedding)
#                     # v_vn = self.vn_v[layer](virtualnode_embedding)
#                     k_vn, v_vn = virtualnode_embedding,virtualnode_embedding
                    
#                     att = (((q * torch.ones_like(q)) * (k * torch.ones_like(k)))* (1.0 / math.sqrt(k.size(-1)))).sum(dim=-1)
#                     att = F.sigmoid(att).unsqueeze(-1)
#                     # att = softmax(att,batch,None,None)
#                     v = att*v
#                     virtualnode_embedding_temp = scatter_add(v,batch,dim=0)
                    
#                     att_vn = (((q_vn * torch.ones_like(q_vn)) * (k_vn * torch.ones_like(k_vn)))* (1.0 / math.sqrt(k_vn.size(-1)))).sum(dim=-1)
#                     att_vn = F.sigmoid(att_vn).unsqueeze(-1)
#                     v_vn = att_vn*v_vn
#                     virtualnode_embedding_temp += v_vn
                    
                    # virtualnode_embedding_temp = self.global_attention(h_list[layer],batch,layer)
                    virtualnode_embedding_temp = self.gmt[layer](h_list[layer],batch)
                    
                    # virtualnode_embedding_temp = global_attention_pool(h_list[layer], batch) + virtualnode_embedding
                else:
                    virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                    # virtualnode_embedding_temp = global_mask_add_pool(h_list[layer], batch, batched_data.is_att) + virtualnode_embedding
                ### transform virtual nodes using MLP
                
                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training=self.training
                    )
                else:
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training=self.training
                    )
                
                # #if no virtual_node_mlp used
                # virtualnode_embedding = virtualnode_embedding + virtualnode_embedding_temp
                
                #for replacing the position of virtual node mlp
                # virtualnode_embedding_temp2 = virtualnode_embedding
                # virtualnode_embedding = virtualnode_embedding_temp

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]
        elif self.JK == "cat":
            node_representation = torch.cat([h_list[0], h_list[-1]], dim=-1)

        return node_representation


def GNNNodeEmbedding(config,virtual_node, *args, **kwargs):
    if virtual_node:
        return GNN_node_Virtualnode(config,*args, **kwargs)
    else:
        return GNN_node(config,*args, **kwargs)
