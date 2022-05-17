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

import numpy as np
import numpy, scipy.sparse
import scipy.sparse as sp


def get_adj_matrix(edge_index_fea, N):
    adj = torch.zeros([N, N])
    adj[edge_index_fea[0, :], edge_index_fea[1, :]] = 1
    Asp = scipy.sparse.csr_matrix(adj)
    Asp = Asp + Asp.T.multiply(Asp.T > Asp) - Asp.multiply(Asp.T > Asp)
    Asp = Asp + sp.eye(Asp.shape[0])

    D1_ = np.array(Asp.sum(axis=1))**(-0.5)
    D2_ = np.array(Asp.sum(axis=0))**(-0.5)
    D1_ = sp.diags(D1_[:,0], format='csr')
    D2_ = sp.diags(D2_[0,:], format='csr')
    A_ = Asp.dot(D1_)
    A_ = D2_.dot(A_)
    # A_ = sparse_mx_to_torch_sparse_tensor(A_)
    return A_

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
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
    for x in tqdm(data, "assembling adj_list result", total=len(data), leave=False, disable=not accelerator.is_main_process if accelerator else False):

        n=x.num_nodes
        
        # adj_list = l
        # ####this is for gnns might have to change for transformers. transformers need the self edge.
        # adj_list[torch.eye(n).bool()] = 1
        # adj_list = adj_list[adj_list!=1]
        # x["weight"] = adj_list
        if n > 1000:
            continue
        x["N"] = torch.tensor([n*n+0.0])
        x["E"] = torch.tensor([len(x.edge_index[0])])
        out_data.append(x)
    return out_data

def combine_results_test(data, adj_list,args,accelerator = None,device = None):
    out_data = []
    result=[]
    # max_len = args.max_input_len
    count=0
    for x in tqdm(data, "assembling adj_list result", total=len(data), leave=False, disable=not accelerator.is_main_process if accelerator else False):

        # result = compute_adjacency_list([x],args)
        if args.gnn_att_node:
            n=x.num_nodes
            if n > 1000:
                continue
            if 'code' in args.dataset:
                att_node_depth = torch.tensor([[21]])
                x['node_depth'] = torch.cat((x['node_depth'],att_node_depth))
                att_node_features = torch.tensor([[-1,-1]])
                att_edge_attr = torch.tensor([[-1,-1]])
            elif 'molpcba' in args.dataset:
                att_node_features = (torch.zeros((1,9)) - 1).long()
                att_edge_attr = (torch.zeros((1,3)) - 1).long()

            x['x'] = torch.cat((x['x'],att_node_features))
            is_att = torch.zeros((n+1))
            is_att[-1] = 1.0
            x["is_att"] = is_att.reshape((-1,1)).long().bool()

            att2nodes = torch.stack((torch.zeros(n+1)+n , torch.arange(n+1)))
            nodes2att = torch.stack((torch.arange(n),torch.zeros(n)+n))
            x['edge_index'] = torch.cat((x['edge_index'],att2nodes,nodes2att),dim=1).to(x['edge_index'].dtype)
            is_att_edge = torch.zeros((len(x['edge_index'])))
            is_att_edge[-(len(att2nodes)+len(nodes2att)):] = 1.0
            x["is_att_edge"] = is_att_edge.reshape((-1,1)).long().bool()

            x['edge_attr'] = torch.cat((x['edge_attr'],att_edge_attr.repeat((n+n+1,1))),dim=0).to(x['edge_attr'].dtype)
            x.num_nodes = n+1
            
        if args.hig:
            if adj_list is not None:
                res = adj_list[len(out_data)]
            else:
                res =  get_adj_matrix(x["edge_index"],x.num_nodes)
            result.append(res)
            x['adj'] = res
        
        out_data.append(x)
        # count+=1
        # if count == 1000:
        #     break
    return out_data, result

def compute_adjacency_list_cached(data, key, root, args, accelerator, device):
    cachefile = f"{root}/OGB_ADJ_{key}.pickle"
    # if os.path.exists(cachefile):
    #     with open(cachefile, "rb") as cachehandle:
    #         logger.debug("using cached result from '%s'" % cachefile)
    #         result = pickle.load(cachehandle)
    #     out_data, _ = combine_results_test(data, result,args,accelerator,device)
    #     return out_data
    return data
    accelerator.wait_for_everyone()
    # return combine_results_test(data, None,args,accelerator, device)
    # return data
    # return combine_results_test(data, None,args,accelerator,device)

    out_data, result = combine_results_test(data, None,args,accelerator,device)
    
    if accelerator.is_main_process:
        with open(cachefile, "wb") as cachehandle:
            logger.debug("saving result to cache '%s'" % cachefile)
            pickle.dump(result, cachehandle)
        logger.info("Got adjacency list data for key %s" % key)
    accelerator.wait_for_everyone()
    return out_data

def a():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj,'data') and torch.is_tensor(obj.data)):
                print(type(obj),obj.size(),obj.device)
        except:
            pass