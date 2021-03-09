import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models.gnn as gnn

class TransformerNodeEncoder(nn.Module):
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group('transformer')
        group.add_argument('--d_model', type=int, default=128, help='transformer d_model. Currently should be equal to emb_dim')
        group.add_argument('--nhead', type=int, default=4, help='transformer heads')
        group.add_argument('--dim_feedforward', type=int, default=512, help='transformer feedforward dim')
        group.add_argument('--transformer_dropout', type=float, default=0.3)
        group.add_argument('--transformer_activation', type=str, default='relu')
        group.add_argument('--num_encoder_layers', type=int, default=4)
        group.add_argument('--max_input_len', default=1000, help='The max input length of transformer input')
        group.add_argument('--transformer_norm_input', action='store_true', default=False)

    def __init__(self, args):
        super().__init__()
        
        # Creating Transformer Encoder Model
        encoder_layer = nn.TransformerEncoderLayer(args.d_model, args.nhead, args.dim_feedforward, args.transformer_dropout, args.transformer_activation)
        encoder_norm = nn.LayerNorm(args.d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, args.num_encoder_layers, encoder_norm)
        self.max_input_len = args.max_input_len

        self.norm_input = None
        if args.transformer_norm_input:
            self.norm_input = nn.LayerNorm(args.d_model)
        self.cls_embedding = None
        if args.graph_pooling == 'cls':
            self.cls_embedding = nn.Parameter(torch.randn([1, 1, args.d_model], requires_grad=True))

    def forward(self, h_node, batch):
        """
            batch: (B * n_b): [0, 0, 1, 1, 1, 2]
            h_node: (B * n_b) x h_d
        """

        padded_h_node, src_padding_mask = _pad_batch(h_node, batch, self.max_input_len) # Pad in the front
        # (S, B, h_d), (B, S)

        if self.cls_embedding is not None:
            expand_cls_embedding = self.cls_embedding.expand(1, padded_h_node.size(1), -1)
            padded_h_node = torch.cat([padded_h_node, expand_cls_embedding], dim=0)

            zeros = src_padding_mask.data.new(src_padding_mask.size(0), 1).fill_(0)
            src_padding_mask = torch.cat([src_padding_mask, zeros], dim=1)
        if self.norm_input is not None:
            padded_h_node = self.norm_input(padded_h_node)

        transformer_out = self.transformer(padded_h_node, src_key_padding_mask=src_padding_mask) # (S, B, h_d)

        return transformer_out, src_padding_mask

def _pad_batch(h_node, batch, max_input_len):
    num_batch = batch[-1] + 1
    num_nodes = []
    masks = []
    for i in range(num_batch):
        mask = batch.eq(i)
        masks.append(mask)
        num_node = mask.sum()
        num_nodes.append(num_node)

    # print(max(num_nodes))
    max_num_nodes = min(max(num_nodes), max_input_len)
    padded_h_node = h_node.data.new(max_num_nodes, num_batch, h_node.size(-1)).fill_(0)
    src_padding_mask = h_node.data.new(num_batch, max_num_nodes).fill_(0).bool()

    for i, mask in enumerate(masks):
        num_node = num_nodes[i]
        if num_node > max_num_nodes:
            num_node = max_num_nodes
        padded_h_node[-num_node:, i] = h_node[mask][-num_node:]
        src_padding_mask[i, :max_num_nodes - num_node] = True # [b, s]
        
    return padded_h_node, src_padding_mask