import math

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.utils import degree


### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim: int, edge_encoder_cls):
        """
        emb_dim (int): node embedding dimensionality
        """
        super(GINConv, self).__init__(aggr="add")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim), torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        # edge_attr is two dimensional after augment_edge transformation
        self.edge_encoder = edge_encoder_cls(emb_dim)
    def forward(self, x, edge_index, edge_attr, edge_index_new=None, weight=None):
        edge_embedding = self.edge_encoder(edge_attr)

        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out
    def message(self, x_j, edge_attr):

        return F.relu(x_j + edge_attr)
    def update(self, aggr_out):
        return aggr_out


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim, edge_encoder_cls):
        super(GCNConv, self).__init__(aggr="add")

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

        # edge_attr is two dimensional after augment_edge transformation
        self.edge_encoder = edge_encoder_cls(emb_dim)

    def forward(self, x, edge_index, edge_attr, edge_index_new=None, weight=None):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index
        # print(edge_index,edge_index.shape)
        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        # print(x.size(0),x.size())
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        # print(deg)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        # print(deg_inv_sqrt)
        
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # print("norm shape",norm.shape)
        # print("x_shape",x.shape)
        # print(norm)
        
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1.0 / deg.view(
            -1, 1
        )
    def message(self, x_j, edge_attr, norm):
        # print(x_j,edge_attr,norm)
        # print(x_j.shape,norm.shape,edge_attr)
        # exit()
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

        # return self.propagate(edge_index_new, x=x, weight=weight) + F.relu(x + self.root_emb.weight) * 1.0
    # def message(self, x_j, weight):
    #     # print(x_j,weight)
    #     # print(x_j.shape,weight.shape)
    #     # exit()
    #     return torch.pow(3,weight.view(-1, 1)+0.0) * F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out

    
from typing import Optional, Tuple, Union
from torch.nn import PReLU, Linear, Sequential, Tanh, ReLU, ELU, BatchNorm1d as BN
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Linear
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
)
from torch_geometric.utils import  remove_self_loops, softmax
from torch_scatter import scatter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import degree as node_degree


class GATConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_encoder_cls = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.fill_value = fill_value

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        self.lin_src = Linear(in_channels, heads * out_channels,
                              bias=False)
        self.lin_dst = self.lin_src

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_encoder_cls is not None:
            self.lin_edge = edge_encoder_cls(heads * out_channels)
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        # if self.lin_edge is not None:
        #     self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.
        edge_attr = self.lin_edge(edge_attr)
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        # alpha = self.edge_update(edge_index, alpha=alpha, edge_attr=edge_attr)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size, edge_attr = edge_attr)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias


        return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:

        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            # edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
    
def add_self_loops(
        edge_index: Tensor, edge_attr: OptTensor = None,
        fill_value: Union[float, Tensor, str] = None,
        num_nodes: Optional[int] = None) -> Tuple[Tensor, OptTensor]:

    N = num_nodes

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_attr is not None:
        if fill_value is None:
            loop_attr = edge_attr.new_full((N, ) + edge_attr.size()[1:], 1.)

        elif isinstance(fill_value, (int, float)):
            loop_attr = edge_attr.new_full((N, ) + edge_attr.size()[1:],
                                           fill_value)
        elif isinstance(fill_value, Tensor):
            loop_attr = fill_value.to(edge_attr.device, edge_attr.dtype)
            if edge_attr.dim() != loop_attr.dim():
                loop_attr = loop_attr.unsqueeze(0)
            sizes = [N] + [1] * (loop_attr.dim() - 1)
            loop_attr = loop_attr.repeat(*sizes)

        elif isinstance(fill_value, str):
            loop_attr = scatter(edge_attr, edge_index[1], dim=0, dim_size=N,
                                reduce=fill_value)
        else:
            raise AttributeError("No valid 'fill_value' provided")

        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index, edge_attr

class GSANConv(MessagePassing):
    def __init__(self, hidden, edge_encoder_cls, config, **kwargs):
        super(GSANConv, self).__init__(aggr='add', **kwargs)
        self.n_head = config.n_head
        self.hidden = hidden
        self.config = config
        self.fill_value = config.fill_value
        self.normalizer = config.normalizer
        self.feature_mlp = config.feature_mlp
        n_embd = hidden
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = Sequential(nn.Linear(n_embd, n_embd),PReLU())
        
        self.fea_mlp = Sequential(
            Linear(hidden, hidden),
            PReLU(),
            Linear(hidden, hidden)
            ,PReLU())
        self.mlp = Sequential(
            Linear(hidden, hidden),
            PReLU(),
            BN(hidden),
            Linear(hidden, hidden)
            ,PReLU())

        self.norm_k = nn.LayerNorm(n_embd)
        self.norm_q = nn.LayerNorm(n_embd)
        self.norm_v = nn.LayerNorm(n_embd)
        self.edge_encoder = edge_encoder_cls(n_embd)

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.edge_encoder(edge_attr)
        edge_index, edge_attr = remove_self_loops(edge_index,edge_attr)
        # print("num edge_attr", edge_attr.size())
        edge_index, edge_attr = add_self_loops(edge_index,edge_attr, num_nodes=x.size(0),fill_value=self.fill_value)

        deg = node_degree(edge_index[1], x.size(0), dtype=x.dtype)
        deg_sqrt = deg.pow(0.5)
        weight = deg_sqrt[edge_index[1]]
        
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, weight = weight)
        out = self.mlp(out) + out
        return out

    def message(self, x_i, x_j, edge_attr,index,ptr,weight):     
        xe = x_j + edge_attr
        k = self.key(xe)
        q = self.query(x_i)
        v = self.value(xe)

        # k = self.key(self.norm_k(xe))
        # q = self.query(self.norm_q(x_i))
        # v = self.value(self.norm_v(xe))
        
        # k = self.key(x_j)
        # q = self.query(x_i)
        # v = self.value(x_j)
        
        B,hs = x_i.size()
        if edge_attr is not None:
            e_b ,e_hs = edge_attr.size()
        if edge_attr is not None:
            edge_attr = edge_attr.reshape((e_b,self.n_head,e_hs//self.n_head))
        k = k.reshape((B,self.n_head,hs//self.n_head))
        q = q.reshape((B,self.n_head,hs//self.n_head))
        v = v.reshape((B,self.n_head,hs//self.n_head))
        
        att = (((q * torch.ones_like(q)) * (k * torch.ones_like(k)))* (1.0 / math.sqrt(k.size(-1)))).sum(dim=-1)
        if self.normalizer == "tanh":
            aggr_emb = F.tanh(att).unsqueeze(-1)
        elif self.normalizer == "sigmoid":
            aggr_emb = F.sigmoid(att).unsqueeze(-1)
        elif self.normalizer =="softmax":
            aggr_emb = softmax(att, index, ptr, None).unsqueeze(-1)
        elif self.normalizer =="weighted_softmax":
            aggr_emb = (softmax(att, index, ptr, None) * weight.reshape((-1,1))).unsqueeze(-1)
        

        feature2d = aggr_emb*v
        
        feature2d = feature2d.reshape((B,hs))
        
        if self.feature_mlp:
            return self.fea_mlp(feature2d)
        else:
            return feature2d
        


#     def update(self, aggr_out, x):
#         k = self.key(x)
#         q = self.query(x)
#         v = self.value(x)
        
#         B,hs = x.size()
#         k = k.reshape((B,self.n_head,hs//self.n_head))
#         q = q.reshape((B,self.n_head,hs//self.n_head))
#         v = v.reshape((B,self.n_head,hs//self.n_head))
        
        
#         att = (((q * torch.ones_like(q)) * (k * torch.ones_like(k)))* (1.0 / math.sqrt(k.size(-1)))).sum(dim=-1)
#         aggr_emb = F.tanh(att).unsqueeze(-1)
#         feature2d = aggr_emb*v
        
#         feature2d = feature2d.reshape((B,hs))
#         if self.feature_mlp:
#             return aggr_out + self.fea_mlp(feature2d)
#         else:
#             return aggr_out + feature2d


    def __repr__(self):
        return self.__class__.__name__
    

class GSANEAConv(MessagePassing):
    def __init__(self, hidden, edge_encoder_cls, config, **kwargs):
        super(GSANEAConv, self).__init__(aggr='add', **kwargs)
        self.n_head = config.n_head
        self.hidden = hidden
        self.config = config
        self.fill_value = config.fill_value
        self.normalizer = config.normalizer
        self.feature_mlp = config.feature_mlp
        n_embd = hidden
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = Sequential(nn.Linear(n_embd, n_embd),PReLU())
        
        self.key_e = nn.Linear(n_embd, n_embd)
        self.query_e = nn.Linear(n_embd, n_embd)
        self.value_e = Sequential(nn.Linear(n_embd, n_embd),PReLU())
        
        self.fea_mlp = Sequential(
            Linear(hidden, hidden),
            PReLU(),
            Linear(hidden, hidden)
            ,PReLU())
        self.mlp = Sequential(
            Linear(hidden, hidden),
            PReLU(),
            BN(hidden),
            Linear(hidden, hidden)
            ,PReLU())
        self.edge_mlp = Sequential(
            Linear(hidden, hidden),
            PReLU(),
            Linear(hidden, hidden))
        self.edge_norm1 = nn.LayerNorm(hidden)
        self.edge_norm2 = nn.LayerNorm(hidden)
        self.edge_drop1 = nn.Dropout(0.3)
        self.edge_drop2 = nn.Dropout(0.3)

        self.edge_encoder = edge_encoder_cls(n_embd)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr.size(1) != x.size(1):
            edge_attr = self.edge_encoder(edge_attr)
            edge_index, edge_attr = remove_self_loops(edge_index,edge_attr)
            edge_index, edge_attr = add_self_loops(edge_index,edge_attr, num_nodes=x.size(0),fill_value=self.fill_value)       
            
    
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.mlp(out) + out
        
        edge_attr  = self.edge_norm1(self.edge_drop1(self.new_edge_attr) + edge_attr)
        edge_attr = self.edge_norm2(self.edge_drop2(self.edge_mlp(edge_attr)) + edge_attr)
        self.new_edge_attr = None
        return out, edge_index,edge_attr

    def message(self, x_i, x_j, edge_attr,index,ptr):     
        # xe = x_j + edge_attr
        k_j = self.key(x_j)
        k_i = self.key(x_i)
        q_i = self.query(x_i)
        v_j = self.value(x_j)
        v_i = self.value(x_i)
        
        k_e = self.key_e(edge_attr)
        q_e = self.query_e(edge_attr)
        v_e = self.value_e(edge_attr)
        
        # print(k_j.shape,k_e.shape)
        B,hs = x_i.size()
        
        # if edge_attr is not None:
        #     e_b ,e_hs = edge_attr.size()
        # if edge_attr is not None:
        #     edge_attr = edge_attr.reshape((e_b,self.n_head,e_hs//self.n_head))
        # k = k.reshape((B,self.n_head,hs//self.n_head))
        # q = q.reshape((B,self.n_head,hs//self.n_head))
        # v = v.reshape((B,self.n_head,hs//self.n_head))
        
        att = (((q_i * torch.ones_like(q_i)) * ((k_j+k_e) * torch.ones_like(k_j)))* (1.0 / math.sqrt(k_i.size(-1)))).sum(dim=-1)
        if self.normalizer == "tanh":
            aggr_emb = F.tanh(att).unsqueeze(-1)
        elif self.normalizer =="softmax":
            aggr_emb = softmax(att, index, ptr, None).unsqueeze(-1)
        

        feature2d = aggr_emb*(v_j+v_e)     
        
        feature2d = feature2d.reshape((B,hs))
        
        # self.new_edge_attr = self.edge_att(k_i, k_j,k_e,q_e,v_i,v_j,v_e)
        self.new_edge_attr = edge_attr
        if self.feature_mlp:
            return self.fea_mlp(feature2d)
        else:
            return feature2d
        
    def edge_att(self,k_i, k_j, k_e, q_e, v_i, v_j, v_e):
        k = torch.stack((k_i,k_j,k_e),dim=1)
        att = torch.einsum('bh,bmh->bm',q_e,k).softmax(dim=-1)
        v = torch.stack((v_i,v_j,v_e),dim=1)
        out = torch.einsum('bm,bmh->bh',att,v)
        return out
        

    def __repr__(self):
        return self.__class__.__name__
    
class GSANENAConv(MessagePassing):
    def __init__(self, hidden, edge_encoder_cls, config, **kwargs):
        super(GSANENAConv, self).__init__(aggr='add', **kwargs)
        self.n_head = config.n_head
        self.hidden = hidden
        self.config = config
        self.fill_value = config.fill_value
        self.normalizer = config.normalizer
        self.feature_mlp = config.feature_mlp
        n_embd = hidden
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = Sequential(nn.Linear(n_embd, n_embd),PReLU())
        
        self.edge_linear = nn.Linear(3*n_embd, n_embd)
        nn.init.normal_(self.edge_linear.weight,0.0,0.1)
        nn.init.eye_(self.edge_linear.weight[:,-n_embd:])
        
        # self.edge_linear = Sequential(self.edge_linear,PReLU(),nn.LayerNorm(n_embd))
        # self.edge_linear = Sequential(nn.Linear(3*n_embd,3),nn.Linear(3, n_embd),PReLU(),nn.LayerNorm(n_embd))
        
        
        self.fea_mlp = Sequential(
            Linear(hidden, hidden),
            PReLU(),
            Linear(hidden, hidden)
            ,PReLU())
        self.mlp = Sequential(
            Linear(hidden, hidden),
            PReLU(),
            BN(hidden),
            Linear(hidden, hidden)
            ,PReLU())
        self.edge_mlp = Sequential(
            Linear(hidden, hidden),
            PReLU(),
            Linear(hidden, hidden))
        self.edge_norm1 = nn.LayerNorm(hidden)
        self.edge_norm2 = nn.LayerNorm(hidden)
        self.edge_drop1 = nn.Dropout(0.3)
        self.edge_drop2 = nn.Dropout(0.3)

        self.edge_encoder = edge_encoder_cls(n_embd)

    def forward(self, x, edge_index, edge_attr):
        if type(edge_attr) is tuple:
            edge_attr, orig_edge_attr = edge_attr
            orig_edge_index = edge_index
            temp_edge_attr = self.edge_encoder(orig_edge_attr)
            # print(len(edge_index[0]))
            edge_index, temp_edge_attr = remove_self_loops(edge_index,temp_edge_attr)
            edge_index, temp_edge_attr = add_self_loops(edge_index,temp_edge_attr, num_nodes=x.size(0),fill_value=self.fill_value) 
            edge_attr += temp_edge_attr
            # edge_attr = temp_edge_attr
        elif edge_attr.size(1) != x.size(1):
            orig_edge_attr = edge_attr
            orig_edge_index = edge_index
            edge_attr = self.edge_encoder(edge_attr)
            # print(len(edge_index[0]))
            edge_index, edge_attr = remove_self_loops(edge_index,edge_attr)
            # print(len(edge_index[0]))
            edge_index, edge_attr = add_self_loops(edge_index,edge_attr, num_nodes=x.size(0),fill_value=self.fill_value) 
            # print(len(edge_index[0]))
            # orig_edge_attr = edge_attr
            
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.mlp(out) + out
        
        edge_attr = self.new_edge_attr
        edge_attr  = self.edge_norm1(self.edge_drop1(edge_attr) + edge_attr)
        edge_attr = self.edge_norm2(self.edge_drop2(self.edge_mlp(edge_attr)) + edge_attr)
        self.new_edge_attr = None
        edge_attr = (edge_attr,orig_edge_attr)
        return out, orig_edge_index,edge_attr

    def message(self, x_i, x_j, edge_attr,index,ptr):     
        xe = x_j + edge_attr
        k = self.key(xe)
        q = self.query(x_i)
        v = self.value(xe) 
        edge_attr = self.edge_linear(torch.cat((x_i,x_j,edge_attr),dim=-1)) + edge_attr
        self.new_edge_attr = edge_attr
        
        # print(k_j.shape,k_e.shape)
        B,hs = x_i.size()
        
        # if edge_attr is not None:
        #     e_b ,e_hs = edge_attr.size()
        # if edge_attr is not None:
        #     edge_attr = edge_attr.reshape((e_b,self.n_head,e_hs//self.n_head))
        # k = k.reshape((B,self.n_head,hs//self.n_head))
        # q = q.reshape((B,self.n_head,hs//self.n_head))
        # v = v.reshape((B,self.n_head,hs//self.n_head))
        
        att = (((q * torch.ones_like(q)) * ((k) * torch.ones_like(k)))* (1.0 / math.sqrt(k.size(-1)))).sum(dim=-1)
        if self.normalizer == "tanh":
            aggr_emb = F.tanh(att).unsqueeze(-1)
        elif self.normalizer =="softmax":
            aggr_emb = softmax(att, index, ptr, None).unsqueeze(-1)
        

        feature2d = aggr_emb*(v)     
        
        feature2d = feature2d.reshape((B,hs))
        
        # self.new_edge_attr = self.edge_att(k_i, k_j,k_e,q_e,v_i,v_j,v_e)
        # self.new_edge_attr = edge_attr
        if self.feature_mlp:
            return self.fea_mlp(feature2d)
        else:
            return feature2d
        

    def __repr__(self):
        return self.__class__.__name__