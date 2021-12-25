import os
import pickle

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
import torch.nn.functional as F

def make_adj_list(N, edge_index_transposed,args):
    A = np.eye(N)
    for edge in edge_index_transposed:
        A[edge[0], edge[1]] = 1
    # A = np.pad(A,((args.max_input_len - len(A),0),(args.max_input_len-len(A),0)),'constant')
    adj_list = A != 0
    # print(N)
    # print(A)
    # print(adj_list)
    # exit()
    return adj_list


def make_adj_list_wrapper(x,args):
    return make_adj_list(x.num_nodes, x["edge_index"].T, args)


def compute_adjacency_list(data,args,accelerator=None):
    out = []
    for x in tqdm(data, "adjacency list", leave=False, disable=not accelerator.is_main_process if accelerator else False):
        out.append(make_adj_list_wrapper(x,args))
    return out


def combine_results(data, adj_list,args,accelerator = None,device = None):
    out_data = []
    max_len = args.max_input_len
    for x, l in tqdm(zip(data, adj_list), "assembling adj_list result", total=len(data), leave=False, disable=not accelerator.is_main_process if accelerator else False):
        # adj_list = F.pad(torch.tensor(l),(0,1,0,1), 'constant',value=True)
        adj_list = F.pad(torch.tensor(l),(0,1,max_len-len(l),1), 'constant',value=True)
        adj_list = F.pad(adj_list,(max_len-len(l),0,0,0), 'constant',value=False).unsqueeze(dim = 0)
        x["adj_list"] = adj_list
        out_data.append(x)
    return out_data


def compute_adjacency_list_cached(data, key, root, args, accelerator, device):
    cachefile = f"{root}/OGB_ADJLIST_{key}.pickle"
    # if os.path.exists(cachefile):
    #     with open(cachefile, "rb") as cachehandle:
    #         logger.debug("using cached result from '%s'" % cachefile)
    #         result = pickle.load(cachehandle)
    #     return combine_results(data, result,args,accelerator,device)
    accelerator.wait_for_everyone()
    result = compute_adjacency_list(data,args,accelerator)
    if accelerator.is_main_process:
        with open(cachefile, "wb") as cachehandle:
            logger.debug("saving result to cache '%s'" % cachefile)
            pickle.dump(result, cachehandle)
        logger.info("Got adjacency list data for key %s" % key)
    accelerator.wait_for_everyone()
    return combine_results(data, result,args,accelerator,device)
