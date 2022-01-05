import os
import pickle
import gc
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
import torch.nn.functional as F
from data.floyd_warshall import floyd_warshall_fastest, floyd_warshall_fastest_torch
# import data.floyd_warshall.floyd_warshall_fastest as floyd_warshall_fastest
from numpy import inf
# from p_tqdm import p_map, p_umap, p_imap, p_uimap
from multiprocessing import Pool

# def make_adj_list(N, edge_index_transposed,args=None):
#     # edge_index_transposed = torch.tensor(edge_index_transposed).cuda()
#     A = np.eye(N)
#     for edge in edge_index_transposed:
#         A[edge[0], edge[1]] = 1
#     # A = np.pad(A,((args.max_input_len - len(A),0),(args.max_input_len-len(A),0)),'constant')
#     A[A==0] = inf
#     A = A - np.eye(N)
#     # A = floyd_warshall_fastest(A)
#     A = torch.tensor(A).cuda()
#     A = floyd_warshall_fastest_torch(A)
#     A = A.cpu()
#     A[A==0] = 1
#     A = -(A-1)
#     # adj_list = A != 0
#     adj_list = torch.tensor(A,dtype = torch.int16)
#     # print(N)
#     # print(A)
#     # print(adj_list)
#     # exit()
#     return adj_list
def make_adj_list(N, edge_index_transposed,args=None):

    edge_index_transposed = torch.tensor(edge_index_transposed, device='cuda')
    A = torch.eye(N, device = 'cuda')
    C = torch.eye(N, device = 'cuda')
    # else:
    #     edge_index_transposed = torch.tensor(edge_index_transposed, device='cuda')
    #     A = torch.eye(N, device = 'cuda')
    #     C = torch.eye(N, device = 'cuda')
    A[edge_index_transposed[:,1],edge_index_transposed[:,0]] = 1
    # A = np.pad(A,((args.max_input_len - len(A),0),(args.max_input_len-len(A),0)),'constant')
    A[A==0] = float('inf')
    A = A - C
    A = floyd_warshall_fastest_torch(A)
    A[A==0] = 1
    A = -(A-1)
    A[A<-120] = -120.0
    adj_list = A.to(torch.int8).cpu()
    del A
    del edge_index_transposed
    del C
    return adj_list

def make_adj_list_wrapper(x,args=None):
    return make_adj_list(x.num_nodes, x["edge_index"].T, args)


# def compute_adjacency_list(data,args,accelerator=None):
#     out = []
#     i = 0
#     a()
#     for x in tqdm(data, "adjacency list", leave=False, disable=not accelerator.is_main_process if accelerator else False):
#         out.append(make_adj_list_wrapper(x,args))
#         i+=1
#         print('\n\n\n\n')
#         a()
#         if i == 3:
#             exit()
#     return out
def compute_adjacency_list(data,args,accelerator=None):
    out = []
    for x in tqdm(data, "adjacency list", leave=False, disable=not accelerator.is_main_process if accelerator else False):
        out.append(make_adj_list_wrapper(x,args))
    return out



def combine_results(data, adj_list,args,accelerator = None,device = None):
    out_data = []
    # max_len = args.max_input_len
    for x, l in tqdm(zip(data, adj_list), "assembling adj_list result", total=len(data), leave=False, disable=not accelerator.is_main_process if accelerator else False):
        # adj_list = F.pad(torch.tensor(l),(0,1,0,1), 'constant',value=True)
        
        # adj_list = F.pad(torch.tensor(l),(0,1,max_len-len(l),1), 'constant',value=True)
        # adj_list = F.pad(adj_list,(max_len-len(l),0,0,0), 'constant',value=False).unsqueeze(dim = 0)
        n=x.num_nodes
        adj_list = l
        ####this is for gnns might have to change for transformers. transformers need the self edge.
        adj_list[torch.eye(n).bool()] = 1
        adj_list = adj_list[adj_list!=1]
        # adj_list = adj_list.reshape((-1,1))
        # x["adj_list"] = adj_list
        # x["weight"] = adj_list.reshape((-1,)) 

        x["weight"] = adj_list
        #n-1 to avoid self edge
        
        
        # row_index = torch.arange(n).reshape((-1,1)).repeat((1,n-1)).reshape((-1,))
        # col_index = torch.arange(n).repeat((n,)).reshape((n,n))
        # col_index[torch.eye(n).bool()] = -1
        # col_index = col_index[col_index!=-1]
        # x["edge_index_new"] = torch.stack((row_index,col_index))
        
        
        # print(x["edge_index_new"].shape,x["weight"].shape,x.num_nodes)
        # exit()
        N = n*n
        x["N"] = N
        out_data.append(x)
    return out_data

def combine_results_test(data, adj_list,args,accelerator = None,device = None):
    out_data = []
    # max_len = args.max_input_len
    count=0
    for x in tqdm(data, "assembling adj_list result", total=len(data), leave=False, disable=not accelerator.is_main_process if accelerator else False):
        # adj_list = F.pad(torch.tensor(l),(0,1,0,1), 'constant',value=True)
        
        # adj_list = F.pad(torch.tensor(l),(0,1,max_len-len(l),1), 'constant',value=True)
        # adj_list = F.pad(adj_list,(max_len-len(l),0,0,0), 'constant',value=False).unsqueeze(dim = 0)
        
        # adj_list = torch.randint(0,5,(x.num_nodes,x.num_nodes))
        # print(x["x"],x.num_nodes)
        result = compute_adjacency_list([x],args)
        # adj_list = torch.zeros((x.num_nodes,x.num_nodes),dtype=torch.int8)
        # adj_list[:,::2] = 1
        # adj_list[:,::3] = 2
        # adj_list[:,::4] = 4
        n=x.num_nodes
        adj_list = result[0]
        ####this is for gnns might have to change for transformers. transformers need the self edge.
        adj_list[torch.eye(n).bool()] = 1
        adj_list = adj_list[adj_list!=1]
        x["weight"] = adj_list

        
        # row_index = torch.arange(N).reshape((-1,1)).repeat((1,N)).reshape((-1,))
        # col_index = torch.arange(N).repeat((N,))
        # x["edge_index"] = torch.stack((row_index,col_index))
        
        x["N"] = n*n
        out_data.append(x)
        count+=1
        if count == 100:
            break
    return out_data

def compute_adjacency_list_cached(data, key, root, args, accelerator, device):
    cachefile = f"{root}/OGB_ADJLIST_{key}.pickle"
    # if os.path.exists(cachefile):
    #     with open(cachefile, "rb") as cachehandle:
    #         logger.debug("using cached result from '%s'" % cachefile)
    #         result = pickle.load(cachehandle)
    #     return combine_results(data, result,args,accelerator,device)
    # return data
    accelerator.wait_for_everyone()
    # return combine_results_test(data, None,args,accelerator, device)
    
    # result = compute_adjacency_list(data,args,accelerator)
    result = compute_adjacency_list(data,args,accelerator)
    
    # if accelerator.is_main_process:
    #     with open(cachefile, "wb") as cachehandle:
    #         logger.debug("saving result to cache '%s'" % cachefile)
    #         pickle.dump(result, cachehandle)
    #     logger.info("Got adjacency list data for key %s" % key)
    accelerator.wait_for_everyone()
    return combine_results(data, result,args,accelerator,device)

def a():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj,'data') and torch.is_tensor(obj.data)):
                print(type(obj),obj.size(),obj.device)
        except:
            pass