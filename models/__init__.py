from .gnn import GNN
from .gnn_transformer import GNNTransformer
from .pna import PNANet
from .pna_transformer import PNATransformer
from .transformer import Transformer
from .transformer_gnn import TransformerGNN
from .gnn_bert import GNNBert
from .pnp import Net

def get_model_and_parser(args, parser):
    model_cls = MODELS[args.model_type]
    model_cls.add_args(parser)
    return model_cls


MODELS = {"gnn-bert": GNNBert, "gnn": GNN, "pna": PNANet, "gnn-transformer": GNNTransformer, "transformer": Transformer, "pna-transformer": PNATransformer, "transformer-gnn": TransformerGNN, "appnp":Net}
