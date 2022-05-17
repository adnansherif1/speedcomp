import os
import random
import math
from datetime import datetime

import configargparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from loguru import logger
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau, LambdaLR
from torch_geometric.data import DataLoader
from tqdm import tqdm
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
import utils
from data.adj_list import compute_adjacency_list_cached
from dataset import DATASET_UTILS
from models import get_model_and_parser
from trainers import get_trainer_and_parser
import accelerate
from accelerate import Accelerator
from dataclasses import dataclass
from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator
kwargs_handlers = [accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)]
accelerator = Accelerator(kwargs_handlers=kwargs_handlers)
device = accelerator.device

if accelerator.is_main_process:
    wandb.init(project="graph-aug")
now = datetime.now()
now = now.strftime("%m_%d-%H_%M_%S")
def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch**3 * init_lr / num_batch_warm_up**3

def main():
    # fmt: off
    parser = configargparse.ArgumentParser(allow_abbrev=False,
                                    description='GNN baselines on ogbg-code data with Pytorch Geometrics')
    parser.add_argument('--configs', required=False, is_config_file=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--wandb_run_idx', type=str, default=None)


    parser.add_argument('--data_tmp', type=str, default=None)
    parser.add_argument('--data_root', type=str, default=None)

    parser.add_argument('--dataset', type=str, default="ogbg-code",
                        help='dataset name (default: ogbg-code)')

    parser.add_argument('--aug', type=str, default='baseline',
                        help='augment method to use [baseline|flag|augment]')
                        
    parser.add_argument('--max_seq_len', type=int, default=None,
                        help='maximum sequence length to predict (default: None)')
   
    group = parser.add_argument_group('model')
    group.add_argument('--model_type', type=str, default='gnn', help='gnn|pna|gnn-transformer')
    group.add_argument('--graph_pooling', type=str, default='mean')
    group = parser.add_argument_group('gnn')
    group.add_argument('--gnn_type', type=str, default='gcn')
    group.add_argument('--gnn_virtual_node', action='store_true')
    group.add_argument('--gnn_dropout', type=float, default=0)
    group.add_argument('--gnn_num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    group.add_argument('--gnn_emb_dim', type=int, default=256,
                        help='dimensionality of hidden units in GNNs (default: 256)')
    group.add_argument('--gnn_JK', type=str, default='last')
    group.add_argument('--gnn_residual', action='store_true', default=False)
    group.add_argument('--gnn_expanded', action='store_true', default=False)
    group.add_argument('--gnn_virtual_attention', action='store_true', default=False)
    group.add_argument('--feature_mlp', action='store_true', default=False)
    group.add_argument('--fill_value', type=float, default=0)
    parser.add_argument('--normalizer', type=str, default='tanh')   
    group.add_argument('--n_head', type=int, default=1)
    group.add_argument('--gnn_att_node', action='store_true', default=False)
    group.add_argument('--hig', action='store_true', default=False)
    group.add_argument('--hig_lr', type=float, default=1e-4)
    group.add_argument('--hig_epochs', type=int, default=15)
    group.add_argument('--hig_warmup_epochs', type=int, default=5)
    
    group = parser.add_argument_group('training')
    group.add_argument('--devices', type=str, default="0",
                        help='which gpu to use if any (default: 0)')
    group.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    group.add_argument('--eval_batch_size', type=int, default=None,
                        help='input batch size for training (default: train batch size)')
    group.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    group.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    group.add_argument('--scheduler', type=str, default=None)
    group.add_argument('--pct_start', type=float, default=0.3)
    group.add_argument('--weight_decay', type=float, default=0.0)
    group.add_argument('--grad_clip', type=float, default=None)
    group.add_argument('--lr', type=float, default=0.001)
    group.add_argument('--max_lr', type=float, default=0.001)
    group.add_argument('--runs', type=int, default=10)
    group.add_argument('--test-freq', type=int, default=3) # Every x runs do an eval
    group.add_argument('--start-eval', type=int, default=2) # Start evaluating in epoch x
    group.add_argument('--resume', type=str, default=None)
    group.add_argument('--seed', type=int, default=None)

    group.add_argument('--warmup_epochs', type=int, default=0)
    # fmt: on

    args, _ = parser.parse_known_args()

    dataset_util = DATASET_UTILS[args.dataset]()
    dataset_util.add_args(parser)
    args, _ = parser.parse_known_args()

    # Setup Trainer and add customized args
    trainer = get_trainer_and_parser(args, parser)
    train = trainer.train
    model_cls = get_model_and_parser(args, parser)
    args = parser.parse_args()
    data_transform = trainer.transform(args)

    run_name = f"{args.dataset}+{model_cls.name(args)}"
    run_name += f"+{trainer.name(args)}+lr={args.lr}+wd={args.weight_decay}"
    if args.scheduler is not None:
        run_name = run_name + f"+sch={args.scheduler}"
    if args.seed:
        run_name = run_name + f"+seed{args.seed}"
    if args.wandb_run_idx is not None:
        run_name = args.wandb_run_idx + "_" + run_name
    if accelerator.is_main_process:
        wandb.run.name = run_name

    # device = torch.device("cuda") if torch.cuda.is_available() and args.devices else torch.device("cpu")
    # device = torch.device("cpu")
    args.save_path = f"exps/{run_name}-{now}"
    os.makedirs(args.save_path, exist_ok=True)
    if args.resume is not None:
        args.save_path = args.resume
    if accelerator.is_main_process:
        logger.info(args)
        wandb.config.update(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if device == torch.cuda.is_available():
            # cudnn.deterministic = True
            torch.cuda.manual_seed(args.seed)
    
    if 'pcqm' in args.dataset:
        dataset_ = PygPCQM4Mv2Dataset(root = '/data/adnans')
        dataset_eval_ = PygPCQM4Mv2Dataset(root = '/data/adnans')
        num_tasks = 1
        node_encoder_cls = lambda: AtomEncoder(model_cls.get_emb_dim(args))
        edge_encoder_cls = lambda emb_dim: BondEncoder(emb_dim=emb_dim)
        evaluator = PCQM4Mv2Evaluator()
        task_type = 'regression'
        split_idx = dataset_.get_idx_split()
        split_idx['test'] = split_idx['test-challenge']
        dataset_.eval_metric = "mae"
        # dataset_[dataset_.get_idx_split()["test"]] = dataset_[dataset_.get_idx_split()["test-dev"]]
        
    elif "ogb" in args.dataset:
        # automatic dataloading and splitting
        dataset_ = PygGraphPropPredDataset(name=args.dataset, root=args.data_root, transform=data_transform)
        dataset_eval_ = PygGraphPropPredDataset(name=args.dataset, root=args.data_root)
        num_tasks, node_encoder_cls, edge_encoder_cls, deg = dataset_util.preprocess(dataset_, dataset_eval_, model_cls, args)
        evaluator = Evaluator(args.dataset)  # automatic evaluator. takes dataset name as input
        task_type = dataset_.task_type
        split_idx = dataset_.get_idx_split()
    else:
        dataset_, num_tasks, node_encoder_cls, edge_encoder_cls, deg = dataset_util.preprocess(args)
        dataset_eval_ = dataset_
        evaluator = None
        task_type = dataset_.task_type
        split_idx = dataset_.get_idx_split()

    
    
    calc_loss = dataset_util.loss_fn(task_type)
    eval = dataset_util.eval

    def create_loader(dataset, dataset_eval):
        # print(dataset[split_idx["test"]])
        # print(dataset[split_idx["test"]][0])
        test_data = compute_adjacency_list_cached(dataset[split_idx["test"]], key=f"{args.dataset}_test", root=args.data_tmp,args = args,accelerator = accelerator,device=device)
        valid_data = compute_adjacency_list_cached(dataset_eval[split_idx["valid"]], key=f"{args.dataset}_valid", root=args.data_tmp,args = args, accelerator= accelerator,device=device)
        train_data = compute_adjacency_list_cached(dataset[split_idx["train"]], key=f"{args.dataset}_train", root=args.data_tmp,args = args,accelerator = accelerator,device=device)
        logger.debug("Finished computing adjacency list")
        # test_data = dataset[split_idx["test"][:5000]]
        # valid_data = dataset_eval[split_idx["valid"][:5000]]
        # train_data = dataset[split_idx["train"][:8000]]

        # print("train set length" , len(train_data))
        # print("half train data", len(train_data[:int(len(train_data)/2)]))
        # print(vars(valid_data))
        # maximum = ""
        # print("ylength", len(valid_data.data.y))
        # for i in valid_data.data.y[0]:
        #     if len(i.split(" ")) > 1:
        #         print(i)
        # print("maximum" , maximum)
        # for i in valid_data.data.y:
        #     print("len",len(i))
        # # if 1:
        # #     return
        eval_bs = args.batch_size if args.eval_batch_size is None else args.eval_batch_size
        # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        # train_loader_eval = DataLoader(train_data, batch_size=eval_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        # valid_loader = DataLoader(valid_data, batch_size=eval_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        # test_loader = DataLoader(test_data, batch_size=eval_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True,persistent_workers=True)
        train_loader_eval = DataLoader(train_data, batch_size=eval_bs, shuffle=False, num_workers=1, pin_memory=True,persistent_workers=True)
        valid_loader = DataLoader(valid_data, batch_size=eval_bs, shuffle=False, num_workers=1, pin_memory=True,persistent_workers=True)
        test_loader = DataLoader(test_data, batch_size=eval_bs, shuffle=False, num_workers=1, pin_memory=True,persistent_workers=True)
        return train_loader, train_loader_eval, valid_loader, test_loader

    train_loader_, train_loader_eval_, valid_loader_, test_loader_ = create_loader(dataset_, dataset_eval_)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    def run(run_id):
        vals, tests = [],[]
        if "ogb" not in args.dataset:
            dataset, _, _, _, _ = dataset_util.preprocess(args)
            dataset_eval = dataset
            # print(dataset_)
            # exit()
            train_loader, train_loader_eval, valid_loader, test_loader = create_loader(dataset, dataset_eval)
        else:
            train_loader, train_loader_eval, valid_loader, test_loader = train_loader_, train_loader_eval_, valid_loader_, test_loader_
            dataset = dataset_
        node_encoder = node_encoder_cls()
        print("training loader length" , len(train_loader))
        os.makedirs(os.path.join(args.save_path, str(run_id)), exist_ok=True)
        best_val, final_test = 0, 0
        #model = model_cls(num_tasks=num_tasks, args=args, node_encoder=node_encoder, edge_encoder_cls=edge_encoder_cls).to(device)
        model = model_cls(num_tasks=num_tasks, args=args, node_encoder=node_encoder, edge_encoder_cls=edge_encoder_cls)
        print("Model Parameters: ", count_parameters(model))
        # exit(-1)
        # model = nn.DataParallel(model)
        if accelerator.is_main_process:
            wandb.watch(model)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        train_loader_eval = accelerator.prepare(train_loader_eval)
        if args.scheduler == "plateau":
            # NOTE(ajayjain): For Molhiv config, this min_lr is too high -- means that lr does not decay.
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, min_lr=0.0001, verbose=False)
        elif args.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=(args.epochs-args.warmup_epochs) * len(train_loader), verbose=False)
        elif args.scheduler == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=args.max_lr,
                epochs=args.epochs,
                steps_per_epoch=len(train_loader),
                pct_start=args.pct_start,
                verbose=False,
            )
        elif args.scheduler == "custom":
            
            def lr_lambda(current_step):
                num_training_steps = args.epochs * len(train_loader) # total number of training steps
                num_warmup_steps = args.warmup_epochs * len(train_loader) # number of warmup steps
                num_fixed_steps = args.unfreeze_bert * len(train_loader) # number of fixed lr steps (for gnn)
                
                if current_step < num_fixed_steps:
                    return 1
                
                if current_step < num_warmup_steps + num_fixed_steps and current_step >= num_fixed_steps:
                    return float(current_step - num_fixed_steps) / (2*float(max(1, num_warmup_steps)))
                
                progress = float(current_step - num_warmup_steps - num_fixed_steps) / float(max(1, num_training_steps - num_warmup_steps - num_fixed_steps))

                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(0.5) * 2.0 * progress)))/2

            scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)
            
        elif args.scheduler is None:
            scheduler = None

        else:
            raise NotImplementedError

        # Load resume model, if any
        
        start_epoch = 1
        last_model_path = os.path.join(args.save_path, str(run_id), "best_model.pt")
        print(last_model_path)
        if os.path.exists(last_model_path):
            state_dict = torch.load(last_model_path)
            start_epoch = state_dict["epoch"] + 1
            model.load_state_dict(state_dict["model"])
            optimizer.load_state_dict(state_dict["optimizer"])
            if args.scheduler:
                scheduler.load_state_dict(state_dict["scheduler"])
            logger.info("[Resume] Loaded: {last_model_path} epoch: {start_epoch}")
            print("everything loaded and ready to go")
            run_id+=1
            
        
        # if args.model_type == "gnn-bert":
        #     model.epoch_callback(epoch=start_epoch - 1)
        
        for epoch in range(start_epoch, args.epochs + 1):
            
            if epoch <= args.warmup_epochs:
                warm_up_lr(epoch, args.warmup_epochs, args.lr, optimizer)
                
            if accelerator.is_main_process:
                logger.info(f"=====Epoch {epoch}=====")
                logger.info("Training...")
                logger.info("Total parameters: {}", utils.num_total_parameters(model))
                logger.info("Trainable parameters: {}", utils.num_trainable_parameters(model))
            loss = train(model, device, train_loader, optimizer, args, calc_loss, scheduler if (args.scheduler != "plateau" and epoch > args.warmup_epochs) else None, accelerator = accelerator)


                
            #accelerator.wait_for_everyone()
            # if args.model_type == "gnn-bert":
            #     model.epoch_callback(epoch)
            if accelerator.is_main_process:
                wandb.log({f"train/loss-runs{run_id}": loss, f"train/lr": optimizer.param_groups[0]["lr"], f"epoch": epoch})

            if args.scheduler == "plateau":
                valid_perf = eval(model, device, valid_loader, evaluator,accelerator,False)
                accelerator.wait_for_everyone()
                
                valid_metric = valid_perf[dataset.eval_metric]

                scheduler.step(valid_metric)
            if True or epoch > args.start_eval and epoch % args.test_freq == 0 or epoch in [1, args.epochs]:##changed
                if accelerator.is_main_process:
                    logger.info("Evaluating...")
                with torch.no_grad():
                    # train_perf = eval(model, device, train_loader_eval, evaluator, accelerator,True)
                    if dataset.eval_metric == "F1":
                        train_perf = {"F1":0,'f1':0}##changed
                    elif dataset.eval_metric == "ap":
                        train_perf = {"ap":0,'ap':0}
                    elif dataset.eval_metric == "mae":
                        train_perf = {"mae":0,'Mae':0}
                    elif dataset.eval_metric == "rocauc":
                        train_perf = {'rocauc':0,'rocauc':0}
                    else:
                        train_perf = {"acc":0,'Acc':0}
                    if args.scheduler != "plateau":
                        valid_perf = eval(model, device, valid_loader, evaluator,accelerator,False)
                    test_perf = eval(model, device, test_loader, evaluator,accelerator,False)
                    accelerator.wait_for_everyone()
                train_metric, valid_metric, test_metric = (
                    train_perf[dataset.eval_metric],
                    valid_perf[dataset.eval_metric],
                    test_perf[dataset.eval_metric],
                )
                if accelerator.is_main_process:
                    wandb.log(
                        {
                            f"train/{dataset.eval_metric}-runs{run_id}": train_metric,
                            f"valid/{dataset.eval_metric}-runs{run_id}": valid_metric,
                            f"test/{dataset.eval_metric}-runs{run_id}": test_metric,
                            "epoch": epoch,
                        }
                    )
                    logger.info(f"Running: {run_name} (runs {run_id})")
                    logger.info(f"Run {run_id} - train: {train_metric}, val: {valid_metric}, test: {test_metric}")
                    
                vals.append(valid_metric)
                tests.append(test_metric)
                
                # Save checkpoints
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                state_dict = {"model": unwrapped_model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
                # state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
                state_dict["scheduler"] = scheduler.state_dict() if args.scheduler else None
                # torch.save(state_dict, os.path.join(args.save_path, str(run_id), "last_model.pt"))
                
                if accelerator.is_main_process:
                    logger.info("[Save] Save model: {}", os.path.join(args.save_path, str(run_id), "last_model.pt"))
                if best_val < valid_metric:
                    best_val = valid_metric
                    final_test = test_metric
                    # accelerator.save(state_dict, os.path.join(args.save_path, str(run_id), "best_model.pt"))
                    if accelerator.is_main_process:
                        torch.save(state_dict, os.path.join(args.save_path, str(run_id), "best_model.pt"))
                        wandb.run.summary[f"best/valid/{dataset.eval_metric}-runs{run_id}"] = valid_metric
                        wandb.run.summary[f"best/test/{dataset.eval_metric}-runs{run_id}"] = test_metric
                        logger.info("[Best Model] Save model: {}", os.path.join(args.save_path, str(run_id), "best_model.pt"))
         
        if args.hig:
            model.gnn_node.hig_started = True #change later
            for g in optimizer.param_groups:
                g['lr'] = args.hig_lr
            scheduler = CosineAnnealingLR(optimizer, T_max=(args.hig_epochs-args.hig_warmup_epochs) * len(train_loader), verbose=False)
            for epoch in range(1, args.hig_epochs + 1):

                if epoch <= args.hig_warmup_epochs:
                    warm_up_lr(epoch, args.hig_warmup_epochs, args.hig_lr, optimizer)

                if accelerator.is_main_process:
                    logger.info(f"=====Epoch {epoch}=====")
                    logger.info("Training...")
                    logger.info("Total parameters: {}", utils.num_total_parameters(model))
                    logger.info("Trainable parameters: {}", utils.num_trainable_parameters(model))
                # loss = train(model, device, train_loader, optimizer, args, calc_loss, scheduler if (args.scheduler != "plateau" and epoch > args.warmup_epochs) else None, accelerator = accelerator)
                loss = 1.0

                if accelerator.is_main_process:
                    wandb.log({f"train/loss-runs{run_id}": loss, f"train/lr": optimizer.param_groups[0]["lr"], f"epoch": epoch})

                if args.scheduler == "plateau":
                    valid_perf = eval(model, device, valid_loader, evaluator,accelerator,False)
                    accelerator.wait_for_everyone()

                    valid_metric = valid_perf[dataset.eval_metric]

                    scheduler.step(valid_metric)
                if True or epoch > args.start_eval and epoch % args.test_freq == 0 or epoch in [1, args.epochs]:##changed
                    if accelerator.is_main_process:
                        logger.info("Evaluating...")
                    with torch.no_grad():
                        # train_perf = eval(model, device, train_loader_eval, evaluator, accelerator,True)
                        if dataset.eval_metric == "F1":
                            train_perf = {"F1":0,'f1':0}##changed
                        elif dataset.eval_metric == "ap":
                            train_perf = {"ap":0,'ap':0}
                        elif dataset.eval_metric == "mae":
                            train_perf = {"mae":0,'Mae':0}
                        elif dataset.eval_metric == "rocauc":
                            train_perf = {'rocauc':0,'rocauc':0}
                        else:
                            train_perf = {"acc":0,'Acc':0}
                        if args.scheduler != "plateau":
                            valid_perf = eval(model, device, valid_loader, evaluator,accelerator,False)
                        test_perf = eval(model, device, test_loader, evaluator,accelerator,False)
                        accelerator.wait_for_everyone()
                    train_metric, valid_metric, test_metric = (
                        train_perf[dataset.eval_metric],
                        valid_perf[dataset.eval_metric],
                        test_perf[dataset.eval_metric],
                    )
                    if accelerator.is_main_process:
                        wandb.log(
                            {
                                f"train/{dataset.eval_metric}-runs{run_id}": train_metric,
                                f"valid/{dataset.eval_metric}-runs{run_id}": valid_metric,
                                f"test/{dataset.eval_metric}-runs{run_id}": test_metric,
                                "epoch": epoch,
                            }
                        )
                        logger.info(f"Running: {run_name} (runs {run_id})")
                        logger.info(f"Run {run_id} - train: {train_metric}, val: {valid_metric}, test: {test_metric}")

                    vals.append(valid_metric)
                    tests.append(test_metric)

        state_dict = torch.load(os.path.join(args.save_path, str(run_id), "best_model.pt"))
        if accelerator.is_main_process:
            logger.info("[Evaluate] Loaded from {}", os.path.join(args.save_path, str(run_id), "best_model.pt"))
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(state_dict["model"])
        best_valid_perf = eval(model, device, valid_loader, evaluator,accelerator,False)
        best_test_perf = eval(model, device, test_loader, evaluator,accelerator,False)
        return best_valid_perf[dataset.eval_metric], best_test_perf[dataset.eval_metric]
        best = 0
        maximum = max(vals)
        for i in range(len(vals)):
            if vals[i] == maximum:
                best = max(best,tests[i])
        return maximum, best

    # print(args)
    vals, tests = [], []
    for run_id in range(args.runs):
        best_val, final_test = run(run_id)
        vals.append(best_val)
        tests.append(final_test)
        if accelerator.is_main_process:
            logger.info(f"Run {run_id} - val: {best_val}, test: {final_test}")
    if accelerator.is_main_process:
        logger.info(f"Average val accuracy: {np.mean(vals)} ± {np.std(vals)}")
        logger.info(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests)}")


if __name__ == "__main__":
    main()

