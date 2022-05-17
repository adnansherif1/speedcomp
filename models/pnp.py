import torch
import torch.nn as nn
import torch.nn.functional as F
from models.operations import APPNP, make_degree, ConvBlock, make_multihop_edges, BondEncoder, OGBMolEmbedding, GlobalPool

class JumpingKnowledge(torch.nn.Module):
    
    def __init__(self, mode):
        super(JumpingKnowledge, self).__init__()
        self.mode = mode.lower()
        #assert self.mode in ['cat']

    def forward(self, xs):
        assert isinstance(xs, list) or isinstance(xs, tuple)
        if self.mode != 'last':
            return torch.cat(xs, dim=-1)
        else:
            return xs[-1]

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.mode)


class Net(torch.nn.Module):
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
    
    def __init__(self,
                 num_tasks,
                 node_encoder, 
                 edge_encoder_cls,
                 args):
        super(Net, self).__init__()
        self.config = args
        self.model = Model(args,node_encoder, edge_encoder_cls,num_tasks)
        print (self.model) 

    def forward(self, batched_data,perturb=None):
        return self.model(batched_data, perturb)

    def __repr__(self):
        return self.__class__.__name__


class Model(nn.Module):
    def __init__(self, config, node_encoder, edge_encoder_cls, num_tasks, degree=True, k =3,T=1):
        super(Model, self).__init__()
        config.appnp = 'true'
        self.config = config
        virtual_node = config.gnn_virtual_node
        self.k = k
        self.T=T
        self.conv_type = config.gnn_type
        hidden = config.gnn_emb_dim
        layers = config.gnn_num_layer
        out_dim = num_tasks
        self.degree = degree
        self.n_head = config.n_head
        convs = [ConvBlock(hidden,
                           nhead = self.n_head,
                           dropout=config.gnn_dropout,
                           virtual_node=virtual_node,
                           k=min(i + 1, self.k),
                           conv_type=self.conv_type,
                           edge_embedding=BondEncoder(emb_dim=hidden))
                 for i in range(layers - 1)]
        convs.append(ConvBlock(hidden,
                               nhead = self.n_head,
                               dropout=config.gnn_dropout,
                               virtual_node=virtual_node,
                               virtual_node_agg=False,  # on last layer, use but do not update virtual node
                               last_layer=True,
                               k=min(layers, self.k),
                               conv_type=self.conv_type,
                               edge_embedding=BondEncoder(emb_dim=hidden)))
        self.main = nn.Sequential(
            OGBMolEmbedding(hidden, embed_edge=False, 
                x_as_list=(self.conv_type=='gin+' or self.conv_type=='gin++'), degree=self.degree),
            *convs)
        if config.appnp == 'true':
            self.aggregate = nn.Sequential(
                APPNP(K=5, alpha=0.8),
                GlobalPool(config.graph_pooling, hidden=hidden),
                nn.Linear(hidden, out_dim)
            )
        else:
            self.aggregate = nn.Sequential(
                GlobalPool(config.graph_pooling, hidden=hidden),
                nn.Linear(hidden, out_dim)
            )
        self.virtual_node = virtual_node
        if self.virtual_node:
            self.v0 = nn.Parameter(torch.zeros(1, hidden), requires_grad=True)

    def forward(self, data, perturb=None):
        # print(data[0])
        data = make_degree(data)
        data = make_multihop_edges(data, self.k)
        if self.virtual_node:
            data.virtual_node = self.v0.expand(data.num_graphs, self.v0.shape[-1])
        data.perturb = perturb
        g = self.main(data)
        if self.conv_type == 'gin+' or self.conv_type == 'gin++':
            g.x = g.x[0]
        if self.training:
            return self.aggregate(g)/self.T
        else:    
            return self.aggregate(g)
