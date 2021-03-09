import torch.nn as nn
from modules.pna_layer import PNAConvSimple
import torch
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from ogb.graphproppred.mol_encoder import AtomEncoder
import torch.nn.functional as F
from .base_model import BaseModel


class PNANet(BaseModel):
    @staticmethod
    def need_deg():
        return True
        
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group('PNANet configs')
        group.add_argument('--aggregators', type=str, nargs='+', default=['mean', 'max', 'min', 'std'])
        group.add_argument('--scalers', type=str, nargs='+', default=['identity', 'amplification', 'attenuation'])
        group.add_argument('--post_layers', type=int, default=1)
        group.add_argument('--add_edge', type=str, default='none')
        group.set_defaults(gnn_residual=True)
        group.set_defaults(gnn_dropout=0.3)
        group.set_defaults(gnn_emb_dim=70)
        group.set_defaults(gnn_num_layer=4)
    
    @staticmethod
    def name(args):
        name = f'{args.model_type}+{args.gnn_type}'
        name += '-virtual' if args.gnn_virtual_node else ''
        return name
        
    def __init__(self, num_tasks, node_encoder, edge_encoder_cls, args):
        super().__init__()
        self.num_layer = args.gnn_num_layer
        self.num_tasks = num_tasks
        self.max_seq_len = args.max_seq_len
        self.aggregators = args.aggregators
        self.scalers = args.scalers
        self.residual = args.gnn_residual
        self.drop_ratio = args.gnn_dropout
        self.graph_pooling = args.graph_pooling

        self.node_encoder = node_encoder

        self.layers = nn.ModuleList(
            [PNAConvSimple(edge_encoder_cls=edge_encoder_cls, add_edge=args.add_edge, in_channels=args.gnn_emb_dim, out_channels=args.gnn_emb_dim, aggregators=self.aggregators, scalers=self.scalers, deg=args.deg, post_layers=args.post_layers, drop_ratio=args.gnn_dropout)
             for _ in range(self.num_layer)])
        self.batch_norms = nn.ModuleList([BatchNorm(args.gnn_emb_dim) for _ in range(self.num_layer)])

        if self.max_seq_len is None:
            self.mlp = nn.Sequential(nn.Linear(args.gnn_emb_dim, 35, bias=True), 
                        nn.ReLU(), 
                        nn.Linear(35, 17, bias=True),
                        nn.ReLU(), 
                        nn.Linear(17, self.num_tasks, bias=True))

        else:
            self.graph_pred_linear_list = torch.nn.ModuleList()
            for i in range(max_seq_len):
                self.graph_pred_linear_list.append(nn.Sequential(
                    nn.Linear(args.gnn_emb_dim, args.gnn_emb_dim),
                    nn.ReLU(), 
                    nn.Linear(args.gnn_emb_dim, self.num_tasks),
                    ))

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(args.gnn_emb_dim, 2*args.gnn_emb_dim), torch.nn.BatchNorm1d(2*args.gnn_emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*args.gnn_emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(args.gnn_emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, batched_data, perturb=None):
        x, edge_index, edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr
        node_depth = batched_data.node_depth if hasattr(batched_data, "node_depth") else None
        encoded_node = self.node_encoder(x) if node_depth is None else self.node_encoder(x, node_depth.view(-1,))
        x = encoded_node + perturb if perturb is not None else encoded_node

        for conv, batch_norm in zip(self.layers, self.batch_norms):
            h = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
            if self.residual:
                x = h + x
            x = F.dropout(x, self.drop_ratio, training=self.training)

        h_graph = self.pool(x, batched_data.batch)

        if self.max_seq_len is None:
            return self.mlp(h_graph)
        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](h_graph))
        return pred_list