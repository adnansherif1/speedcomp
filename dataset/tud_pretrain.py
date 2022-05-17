import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from tqdm import tqdm
from torchvision import transforms

class TUUtil_pretrain:
    @staticmethod
    def add_args(parser):
        parser.set_defaults(batch_size=128)
        parser.set_defaults(epochs=10000)
        parser.set_defaults(lr=0.0005)
        parser.set_defaults(weight_decay=0.0001)
        parser.set_defaults(gnn_dropout=0.5)
        parser.set_defaults(gnn_emb_dim=128)

    @staticmethod
    def loss_fn(task_type):
        def calc_loss(pred, batch, m=1.0):
            #loss = F.nll_loss(pred, batch.y)
            # loss = nn.LogSoftmax(dim=1)(loss) # changed to fix loss function for tud
            loss = F.cross_entropy(pred, batch.y)
            return loss

        return calc_loss

    @staticmethod
    @torch.no_grad()
    def eval(model, device, loader, evaluator,accelerator = None,parallel = False):
        model.eval()

        correct = 0
        for step, batch in enumerate(tqdm(loader, desc="Eval")):
            batch = batch.to(device)

            pred = model(batch)
            pred = pred.max(dim=1)[1]
            correct += pred.eq(batch.y).sum().item()
        return {"acc": correct / len(loader.dataset)}

    @staticmethod
    def preprocess(args, dataset_name):
        dataset = TUDataset(os.path.join(args.data_root, dataset_name), name=dataset_name)
        
        num_tasks = dataset.num_classes

        num_features = dataset.num_features
        if num_features == 0:
            dataset_transform = [add_zeros]
            if dataset.transform is not None:
                dataset_transform.append(dataset.transform)
            dataset.transform = transforms.Compose(dataset_transform)
            num_features = 1
            
        num_training = int(len(dataset) * 0.8)
        num_val = int(len(dataset) * 0.1)
        num_test = len(dataset) - (num_training + num_val)
        training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

        class Dataset(dict):
            pass

        dataset = Dataset({"train": training_set, "valid": validation_set, "test": test_set})
        # print(dataset)
        # print(dataset["train"])
        # print(dataset["train"][0])
        # exit()
        # print(dataset.x)
        dataset.eval_metric = "acc"
        dataset.task_type = "classification"
        dataset.get_idx_split = lambda: {"train": "train", "valid": "valid", "test": "test"}
        # print("features",num_features)
        # exit()
        # node_encoder_cls = lambda: nn.Linear(num_features, args.gnn_emb_dim)
        node_encoder_cls = lambda: lambda x : x

        def edge_encoder_cls(_):
            def zero(_):
                return 0

            return zero

        return dataset, num_tasks, node_encoder_cls, edge_encoder_cls, None

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.float).reshape((-1,1))
    return data