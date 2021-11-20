import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import BertConfig

import math
from modules.gnn_module import GNNNodeEmbedding
from modules.my_bert import MyBertModel
# from modules.masked_transformer_encoder import MaskedOnlyTransformerEncoder
# from modules.transformer_encoder import TransformerNodeEncoder
from modules.utils import pad_batch

from .base_model import BaseModel


class GNNBert(BaseModel):
    @staticmethod
    def get_emb_dim(args):
        return args.gnn_emb_dim

    def add_args(parser):
        # TransformerNodeEncoder.add_args(parser)
        # MaskedOnlyTransformerEncoder.add_args(parser)
        MyBertModel.add_args(parser)
        group = parser.add_argument_group("GNNBert - Training Config")
        group.add_argument("--pos_encoder", default=False, action='store_true')
        group.add_argument("--pretrained_gnn", type=str, default=None, help="pretrained gnn_node node embedding path")
        group.add_argument("--freeze_gnn", type=int, default=None, help="Freeze gnn_node weight from epoch `freeze_gnn`")
        group.add_argument("--freeze_bert", type=int, default=None, help="Freeze my_bert from epoch `freeze_bert")
        group.add_argument("--unfreeze_bert", type=int, default=None, help="Unfreeze my_bert from epoch `unfreeze_bert")
    
    @staticmethod
    def name(args):
        name = f"{args.model_type}-pooling={args.graph_pooling}"
        name += "-norm_input" if args.transformer_norm_input else ""
        name += f"+{args.gnn_type}"
        name += "-virtual" if args.gnn_virtual_node else ""
        name += f"-JK={args.gnn_JK}"
        name += f"-enc_layer={args.num_encoder_layers}"
        # name += f"-enc_layer_masked={args.num_encoder_layers_masked}"
        name += f"-d={args.d_model}"
        name += f"-act={args.transformer_activation}"
        name += f"-tdrop={args.transformer_dropout}"
        name += f"-gdrop={args.gnn_dropout}"
        name += "-pretrained_gnn" if args.pretrained_gnn else ""
        name += f"-freeze_gnn={args.freeze_gnn}" if args.freeze_gnn is not None else ""
        name += f"-freeze_bert={args.freeze_bert}" if args.freeze_bert is not None else ""
        name += f"-unfreeze_bert={args.unfreeze_bert}" if args.unfreeze_bert is not None else ""
        # name += "-prenorm" if args.transformer_prenorm else "-postnorm"
        return name

    def __init__(self, num_tasks, node_encoder, edge_encoder_cls, args):
        super().__init__()
        self.gnn_node = GNNNodeEmbedding(
            args.gnn_virtual_node,
            args.gnn_num_layer,
            args.gnn_emb_dim,
            node_encoder,
            edge_encoder_cls,
            JK=args.gnn_JK,
            drop_ratio=args.gnn_dropout,
            residual=args.gnn_residual,
            gnn_type=args.gnn_type,
        )
        if args.pretrained_gnn:
            # logger.info(self.gnn_node)
            state_dict = torch.load(args.pretrained_gnn)
            state_dict = self._gnn_node_state(state_dict["model"])
            logger.info("Load GNN state from: {}", state_dict.keys())
            self.gnn_node.load_state_dict(state_dict)

        self.freeze_gnn = args.freeze_gnn
        self.freeze_bert = args.freeze_bert
        self.unfreeze_bert = args.unfreeze_bert
        
        gnn_emb_dim = 2 * args.gnn_emb_dim if args.gnn_JK == "cat" else args.gnn_emb_dim
        self.gnn2bert = nn.Linear(gnn_emb_dim, args.d_model)
        # self.pos_encoder = PositionalEncoding(args.d_model, dropout=0) if args.pos_encoder else None
        # self.transformer_encoder = TransformerNodeEncoder(args)
        # self.masked_transformer_encoder = MaskedOnlyTransformerEncoder(args)
        # self.num_encoder_layers = args.num_encoder_layers
        # self.num_encoder_layers_masked = args.num_encoder_layers_masked
        print("Checkpoint is: ", args.checkpoint)
        print("LR is ", args.lr)
        self.checkpoint = args.checkpoint
        self.my_bert_config = BertConfig()
        if self.checkpoint:
            self.my_bert_model = MyBertModel(self.my_bert_config).from_pretrained(self.checkpoint)
        else:
            self.my_bert_model = MyBertModel(self.my_bert_config)
            print("using untrained bert")

        if True:
            self.cls_embedding = nn.Parameter(torch.randn([1, 1, 768], requires_grad=True))

        self.num_tasks = num_tasks
        self.pooling = args.graph_pooling
        self.graph_pred_linear_list = torch.nn.ModuleList()

        self.max_seq_len = args.max_seq_len
        output_dim = args.d_model

        if args.max_seq_len is None:
            self.graph_pred_linear = torch.nn.Linear(output_dim, output_dim)
            self.graph_pred_linear2 = torch.nn.Linear(output_dim, self.num_tasks)
        else:
            for i in range(args.max_seq_len):
                self.graph_pred_linear_list.append(torch.nn.Linear(output_dim, self.num_tasks))
       
        
    def forward(self, batched_data, perturb=None):
        h_node = self.gnn_node(batched_data, perturb)
        h_node = self.gnn2bert(h_node)  # [s, b, d_model]

        padded_h_node, src_padding_mask, num_nodes, mask, max_num_nodes = pad_batch(
            h_node, batched_data.batch, self.my_bert_config.max_position_embeddings, get_mask=True
        )  # Pad in the front
       
        if self.cls_embedding is not None:
            expand_cls_embedding = self.cls_embedding.expand(1, padded_h_node.size(1), -1)
            padded_h_node = torch.cat([expand_cls_embedding, padded_h_node], dim=0)
       
        # print("Padded Node Shape: ", padded_h_node.shape)
        inputs_embeds=padded_h_node.permute(1,0,2)
        # print("Input Embeds Shape: ", inputs_embeds.shape)
        bert_out = self.my_bert_model(inputs_embeds=inputs_embeds) # With CLS at the end gnn_outputs

        h_graph = bert_out.pooler_output # CLS

        if self.max_seq_len is None:
            out = self.graph_pred_linear2(h_graph)
            # out = self.softm(out) # Added to NCI1 loss
            return out
            
        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](h_graph))

        return pred_list

    def epoch_callback(self, epoch):
        # TODO: maybe unfreeze the gnn at the end.
        if self.freeze_gnn is not None and epoch >= self.freeze_gnn: # TODO Ask Paras whether this should be >= or ==
            logger.info(f"Freeze GNN weight after epoch: {epoch}")
            for param in self.gnn_node.parameters():
                param.requires_grad = False

        if self.freeze_bert is not None and epoch == self.freeze_bert:
            logger.info(f"Freeze BERT weight after epoch: {epoch} ")
            for param in self.my_bert_model.parameters():
                param.requires_grad = False

        if self.unfreeze_bert is not None and epoch == self.unfreeze_bert:
            logger.info(f"Unfreeze BERT weight after epoch: {epoch} ")
            for param in self.my_bert_model.parameters():
                param.requires_grad = True
            
    def _gnn_node_state(self, state_dict):
        module_name = "gnn_node"
        new_state_dict = dict()
        for k, v in state_dict.items():
            if module_name in k:
                new_key = k.split(".")
                module_index = new_key.index(module_name)
                new_key = ".".join(new_key[module_index + 1 :])
                new_state_dict[new_key] = v
        return new_state_dict


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)