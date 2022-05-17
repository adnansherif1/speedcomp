import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    # GlobalAttention,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from modules.utils import pad_batch, GlobalAttention, GraphMultisetTransformer
from modules.gnn_module import GNNNodeEmbedding

from .base_model import BaseModel

def global_att_pool(x, is_att, n =None):
    return x[is_att.reshape((-1,))]

class GNN(BaseModel):
    @staticmethod
    def get_emb_dim(args):
        return args.gnn_emb_dim

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("Masked Transformer Encoder -- architecture config")
        group.add_argument("--gat_heads", type=int, default=4)

    @staticmethod
    def name(args):
        name = f"{args.model_type}+{args.gnn_type}"
        name += "-virtual" if args.gnn_virtual_node else ""
        return name

    def __init__(self, num_tasks, node_encoder, edge_encoder_cls, args):
        """
        num_tasks (int): number of labels to be predicted
        virtual_node (bool): whether to add virtual node or not
        """

        super(GNN, self).__init__()
        self.att_node = args.gnn_att_node
        self.num_layer = args.gnn_num_layer
        self.drop_ratio = args.gnn_dropout
        self.JK = args.gnn_JK
        self.emb_dim = args.gnn_emb_dim
        self.num_tasks = num_tasks
        self.max_seq_len = args.max_seq_len
        self.graph_pooling = args.graph_pooling
        self.gat_heads = args.gat_heads
        self.dataset = args.dataset
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_node = GNNNodeEmbedding(
            args,
            args.gnn_virtual_node,
            self.num_layer,
            self.emb_dim,
            node_encoder,
            edge_encoder_cls,
            JK=self.JK,
            drop_ratio=self.drop_ratio,
            residual=args.gnn_residual,
            gnn_type=args.gnn_type,
            gat_heads = self.gat_heads,
            expanded = args.gnn_expanded,
            virtual_attention = args.gnn_virtual_attention
        )

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "gmt":
            self.pool = GraphMultisetTransformer(self.emb_dim,self.emb_dim,self.emb_dim,num_nodes=28)
        elif self.graph_pooling == "global_attention":
            self.pool = GlobalAttention(self.emb_dim,5) 
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(
                    torch.nn.Linear(self.emb_dim, 2 * self.emb_dim),
                    torch.nn.BatchNorm1d(2 * self.emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * self.emb_dim, 1),
                )
            )
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(self.emb_dim, processing_steps=2)
        elif self.graph_pooling == "cls":
            self.pool = global_att_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        if self.max_seq_len is None:
            if self.graph_pooling == "set2set":
                self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)
            else:
                self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear_list = torch.nn.ModuleList()
            if self.graph_pooling == "set2set":
                for i in range(self.max_seq_len):
                    self.graph_pred_linear_list.append(torch.nn.Linear(2 * self.emb_dim, self.num_tasks))
            else:
                for i in range(self.max_seq_len):
                    self.graph_pred_linear_list.append(torch.nn.Linear(self.emb_dim, self.num_tasks))

    def forward(self, batched_data, perturb=None):
        """
        Return:
            A (list of) predictions.
            i-th element represents prediction at i-th position of the sequence.
        """

        h_node = self.gnn_node(batched_data, perturb)
        
        if self.att_node:
            h_graph = self.pool(h_node, batched_data.is_att)
        else:
            h_graph = self.pool(h_node, batched_data.batch)

        if self.max_seq_len is None:
            output =  self.graph_pred_linear(h_graph)
            if 'pcqm' in self.dataset:
                if self.training:
                    return output
                else:
                    return torch.clamp(output, min=0, max=20)
            else:
                return output

        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](h_graph))

        return pred_list


if __name__ == "__main__":
    pass
