import torch
import numpy as np
from loguru import logger
import torch.nn.functional as F

def pad_batch(h_node, batch, max_input_len, get_mask=False):
    num_batch = batch[-1] + 1
    num_nodes = []
    masks = []
    # print(h_node.shape, batch.shape)
    
    for i in range(num_batch):
        mask = batch.eq(i)
        masks.append(mask)
        num_node = mask.sum()
        num_nodes.append(num_node)
    # logger.info(max(num_nodes))
    max_num_nodes = min(max(num_nodes), max_input_len)
    padded_h_node = h_node.data.new(max_num_nodes, num_batch, h_node.size(-1)).fill_(0)
    src_padding_mask = h_node.data.new(num_batch, max_num_nodes).fill_(0).bool()

    for i, mask in enumerate(masks):
        num_node = num_nodes[i]
        if num_node > max_num_nodes:
            num_node = max_num_nodes
        padded_h_node[-num_node:, i] = h_node[mask][-num_node:]
        src_padding_mask[i, : max_num_nodes - num_node] = True  # [b, s]

    if get_mask:
        return padded_h_node, src_padding_mask, num_nodes, masks, max_num_nodes
    return padded_h_node, src_padding_mask

def mask_batch(mask, max_length,nhead,virtual_node):
    attn_mask = mask==False
    attn_mask= attn_mask[:,-(max_length+1):,-(max_length+1):]
    attn_mask = attn_mask.repeat((1,nhead,1)).reshape((-1,max_length+1,max_length+1))
    # attn_mask = torch.tensor(attn_mask).cuda() ##accelerator has to be passed in 
#     for i in range(len(masks[0])):
        
#         # print(attn_mask[0][-2])
#         print(masks[0][i])
#     exit()
    # # attn_mask = attn_mask.repeat((1,nhead,1)).reshape((-1,max_length+1,max_length+1))
    # # attn_mask = attn_mask.repeat((nhead,1,1))
    # attn_mask = torch.zeros((nhead*len(masks),max_length+1,max_length+1), dtype=torch.bool).cuda()
    return attn_mask

def unpad_batch(padded_h_node, prev_h_node, num_nodes, origin_mask, max_num_nodes):
    """
    padded_h_node: [s, b, f]
    prev_h_node: [bxs, f]
    batch: [n]
    pad_mask: [b, s]
    """

    for i, mask in enumerate(origin_mask):
        num_node = num_nodes[i]
        if num_node > max_num_nodes:
            num_node = max_num_nodes
            # cutoff mask
            indices = mask.nonzero()
            indices = indices[-num_node:]
            mask = torch.zeros_like(mask)
            mask[indices] = True
        # logger.info("prev_h_node:", prev_h_node.size())
        # logger.info("padded_h_node:", padded_h_node.size())
        # logger.info("mask:", mask.size())
        prev_h_node = prev_h_node.masked_scatter(mask.unsqueeze(-1), padded_h_node[-num_node:, i])
    return prev_h_node

from typing import Optional

import torch
from torch import Tensor
from torch_scatter import scatter_add

from torch_geometric.utils import softmax




class GlobalAttention(torch.nn.Module):
    def __init__(self, hidden,num_gates,num_layers=1):
        super().__init__()
        self.hidden = hidden
        self.num_gates = num_gates
        self.gate_nn = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor(hidden,num_gates)) for _ in range(num_layers)])
        

        self.reset_parameters()


    def reset_parameters(self):
        for param in self.gate_nn:
            torch.nn.init.xavier_uniform_(param)


    def forward(self, x: Tensor, batch, layer=0) -> Tensor:
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.int64)

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = int(batch.max()) + 1

        gate = x @ self.gate_nn[layer]
        # x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate.mean(dim=-1).reshape((-1,1)) * x, batch, dim=0, dim_size=size)

        return out


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(gate_nn={self.gate_nn}, '
                f'nn={self.nn})')

    
import math
from typing import List, Optional, Tuple, Type

import torch
from torch import Tensor
from torch.nn import LayerNorm, Linear

from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch


class MAB(torch.nn.Module):
    r"""Multihead-Attention Block."""
    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int,
                 Conv: Optional[Type] = None, layer_norm: bool = False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.layer_norm = layer_norm

        self.fc_q = Linear(dim_Q, dim_V)

        if Conv is None:
            self.layer_k = Linear(dim_K, dim_V)
            self.layer_v = Linear(dim_K, dim_V)
        else:
            self.layer_k = Conv(dim_K, dim_V)
            self.layer_v = Conv(dim_K, dim_V)

        if layer_norm:
            self.ln0 = LayerNorm(dim_V)
            self.ln1 = LayerNorm(dim_V)

        self.fc_o = Linear(dim_V, dim_V)

    def reset_parameters(self):
        self.fc_q.reset_parameters()
        self.layer_k.reset_parameters()
        self.layer_v.reset_parameters()
        if self.layer_norm:
            self.ln0.reset_parameters()
            self.ln1.reset_parameters()
        self.fc_o.reset_parameters()
        pass

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:

        Q = self.fc_q(Q)

        if graph is not None:
            x, edge_index, batch = graph
            K, V = self.layer_k(x, edge_index), self.layer_v(x, edge_index)
            K, _ = to_dense_batch(K, batch)
            V, _ = to_dense_batch(V, batch)
        else:
            K, V = self.layer_k(K), self.layer_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), dim=0)
        K_ = torch.cat(K.split(dim_split, 2), dim=0)
        V_ = torch.cat(V.split(dim_split, 2), dim=0)

        if mask is not None:
            mask = torch.cat([mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1, 2))
            attention_score = attention_score / math.sqrt(self.dim_V)
            A = torch.softmax(mask + attention_score, 1)
        else:
            A = torch.softmax(
                Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 1)

        out = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        if self.layer_norm:
            out = self.ln0(out)

        out = out + self.fc_o(out).relu()

        if self.layer_norm:
            out = self.ln1(out)

        return out


class SAB(torch.nn.Module):
    r"""Self-Attention Block."""
    def __init__(self, in_channels: int, out_channels: int, num_heads: int,
                 Conv: Optional[Type] = None, layer_norm: bool = False):
        super().__init__()
        self.mab = MAB(in_channels, in_channels, out_channels, num_heads,
                       Conv=Conv, layer_norm=layer_norm)

    def reset_parameters(self):
        self.mab.reset_parameters()

    def forward(
        self,
        x: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.mab(x, x, graph, mask)


class PMA(torch.nn.Module):
    r"""Graph pooling with Multihead-Attention."""
    def __init__(self, channels: int, num_heads: int, num_seeds: int,
                 Conv: Optional[Type] = None, layer_norm: bool = False):
        super().__init__()
        self.S = torch.nn.Parameter(torch.Tensor(1, num_seeds, channels))
        self.mab = MAB(channels, channels, channels, num_heads, Conv=Conv,
                       layer_norm=layer_norm)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.S)
        self.mab.reset_parameters()

    def forward(
        self,
        x: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.mab(self.S.repeat(x.size(0), 1, 1), x, graph, mask)


class GraphMultisetTransformer(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        Conv: Optional[Type] = None,
        num_nodes: int = 300,
        pooling_ratio: float = 0.25,
        pool_sequences: List[str] = ['GMPool_G', 'SelfAtt', 'GMPool_I'],
        num_heads: int = 4,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.Conv = Conv or GCNConv
        self.num_nodes = num_nodes
        self.pooling_ratio = pooling_ratio
        self.pool_sequences = pool_sequences
        self.num_heads = num_heads
        self.layer_norm = layer_norm

        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

        self.pools = torch.nn.ModuleList()
        num_out_nodes = math.ceil(num_nodes * pooling_ratio)
        for i, pool_type in enumerate(pool_sequences):
            if pool_type not in ['GMPool_G', 'GMPool_I', 'SelfAtt']:
                raise ValueError("Elements in 'pool_sequences' should be one "
                                 "of 'GMPool_G', 'GMPool_I', or 'SelfAtt'")

            if i == len(pool_sequences) - 1:
                num_out_nodes = 1

            if pool_type == 'GMPool_G':
                self.pools.append(
                    PMA(hidden_channels, num_heads, num_out_nodes,
                        Conv=None, layer_norm=layer_norm))
                num_out_nodes = math.ceil(num_out_nodes * self.pooling_ratio)

            elif pool_type == 'GMPool_I':
                self.pools.append(
                    PMA(hidden_channels, num_heads, num_out_nodes, Conv=None,
                        layer_norm=layer_norm))
                num_out_nodes = math.ceil(num_out_nodes * self.pooling_ratio)

            elif pool_type == 'SelfAtt':
                self.pools.append(
                    SAB(hidden_channels, hidden_channels, num_heads, Conv=None,
                        layer_norm=layer_norm))

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()


    def forward(self, x: Tensor, batch: Tensor,
                edge_index: Optional[Tensor] = None) -> Tensor:
        """"""
        x = self.lin1(x)
        batch_x, mask = to_dense_batch(x, batch)
        mask = (~mask).unsqueeze(1).to(dtype=x.dtype) * -1e9

        for i, (name, pool) in enumerate(zip(self.pool_sequences, self.pools)):
            graph =  None
            batch_x = pool(batch_x, graph, mask)
            mask = None

        return self.lin2(batch_x.squeeze(1))


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, pool_sequences={self.pool_sequences})')