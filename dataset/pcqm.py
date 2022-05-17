import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.utils import degree
from tqdm import tqdm
from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator

class PcqmUtil:
    @staticmethod
    def add_args(parser):
        parser.add_argument("--feature", type=str, default="full", help="full feature or simple feature")
        parser.set_defaults(batch_size=32)
        parser.set_defaults(epochs=100)
        parser.set_defaults(gnn_dropout=0.5)

    @staticmethod
    def loss_fn(task_type):
        # cls_criterion = torch.nn.BCEWithLogitsLoss()
        # reg_criterion = torch.nn.MSELoss()
        reg_criterion = torch.nn.L1Loss()

        def calc_loss(pred, batch, m=1.0):
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss /= m
            return loss

        return calc_loss

    @staticmethod
    def eval(model, device, loader, evaluator,accelerator,parallel):
        model.eval()
        y_true = []
        y_pred = []

        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)

            with torch.no_grad():
                pred = model(batch).view(-1,)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim = 0)
        y_pred = torch.cat(y_pred, dim = 0)

        input_dict = {"y_true": y_true, "y_pred": y_pred}

        return evaluator.eval(input_dict)

    @staticmethod
    def preprocess(dataset, dataset_eval, model_cls, args):
        
        return 0,0,0,18
