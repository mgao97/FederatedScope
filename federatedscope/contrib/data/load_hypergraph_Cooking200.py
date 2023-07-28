from federatedscope.core.data.utils import convert_data_mode
from federatedscope.core.auxiliaries.utils import setup_seed

import dhg
import time
import numpy as np
import copy
from copy import deepcopy
import torch
import torch.optim as optim
import torch.nn.functional as F

from dhg import Graph, Hypergraph
from dhg.data import Cooking200
from dhg.models import GCN
from dhg.models import HGNN
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from federatedscope.core.data import BaseDataTranslator
from federatedscope.core.splitters.graph import LouvainSplitter
from federatedscope.register import register_data
from federatedscope.core.data import DummyDataTranslator


def load_cooking200(config):

    # num_split = [232, 542, np.iinfo(np.int64).max]

    hdataset = Cooking200()
    X, lbl = torch.eye(hdataset["num_vertices"]), hdataset["labels"]
    HG = Hypergraph(hdataset['num_vertices'], hdataset['edge_list'])
    G = Graph.from_hypergraph_clique(HG, weighted=True)
    node0, node1, edge_weight = [], [], []
    for edge in G.e[0]:
        node0.append(edge[0])
        node1.append(edge[1])
    for weight in G.e[1]:
        edge_weight.append(weight)
    edge_index = torch.tensor([node0, node1],dtype=int)
    edge_weight = torch.tensor(edge_weight,dtype=float)
    train_mask, val_mask, test_mask = hdataset['train_mask'], hdataset['val_mask'], hdataset['test_mask']
    dataset = Data(x=X, edge_index= edge_index, edge_wight=edge_weight, y=lbl, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    global_data = copy.deepcopy(dataset)

    dataset = LouvainSplitter(config.federate.client_num)(dataset)

    data_local_dict = dict()
    for client_idx in range(len(dataset)):
        local_data = dataset[client_idx]
        data_local_dict[client_idx+1] = {
            'data': local_data,
            'train': [local_data],
            'val': [local_data],
            'test': [local_data]
        }
    
    data_local_dict[0] = {
        'data': global_data,
        'train': [global_data],
        'val': [global_data],
        'test': [global_data]
    }

    translator = DummyDataTranslator(config)
    return translator(data_local_dict), config



def call_cooking200(config, client_cfgs):
    if config.data.type == "hg_cooking200":
        data, modified_config = load_cooking200(config)
        return data, modified_config

register_data('hg_cooking200', call_cooking200)


# def train(net, X, A, lbls, train_idx, optimizer, epoch):
#     net.train()
#     st = time.time()
#     optimizer.zero_grad()
#     outs = net(X, A)
#     outs, lbls = outs[train_idx], lbls[train_idx]
#     loss = F.cross_entropy(outs, lbls)
#     loss.backward()
#     optimizer.step()
#     print(f'Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}')
#     return loss.item()

# @torch.no_grad()
# def infer(net, X, A, lbls, idx, test=False):
#     net.eval()
#     outs = net(X, A)
#     outs, lbls = outs[idx], lbls[idx]
#     if not test:
#         res = evaluator.validate(lbls, outs)
#     else:
#         res = evaluator.test(lbls, outs)
#     return res
    


# if __name__ == '__main__':

#     set_seed(2021)
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
#     data = Cooking200()
#     X, lbl = torch.eye(data["num_vertices"]), data["labels"]
#     HG = Hypergraph(data["num_vertices"], data["edge_list"])
#     G = Graph.from_hypergraph_clique(HG, weighted=True)
#     train_mask, val_mask, test_mask = data['train_mask'], data['val_mask'], data['test_mask']

#     # print some basic information about the dataset
#     print('some info. about the dataset: ............')
#     print(X, X.shape)
#     print(lbl, lbl.shape, torch.unique(lbl))
#     print(G)
#     print(train_mask, train_mask.shape)

#     net = GCN(X.shape[1], 32, data['num_classes'], use_bn=True)
#     optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

#     X, lbl = X.to(device), lbl.to(device)
#     G = G.to(device)
#     net = net.to(device)

#     best_state = None
#     best_epoch, best_val = 0, 0
#     for epoch in range(200):
#         # train
#         train(net, X, G, lbl, train_mask, optimizer, epoch)
#         # validation
#         if epoch % 5 == 0:
#             with torch.no_grad():
#                 val_res = infer(net, X, G, lbl, val_mask, test=False)
#             if val_res > best_val:
#                 print(f'update best: {val_res:.5f}')
#                 best_val = val_res
#                 best_epoch, best_state = epoch, deepcopy(net.state_dict())
#                 torch.save(best_state,'gcn_best_model_params.pt')
#     print('\ntrain finished!')
#     print(f"best val: {best_val:.5f}")
#     # test
#     print("test...")
#     net.load_state_dict(best_state)
#     res = infer(net, X, G, lbl, test_mask, test=True)
#     print(f'final result: epoch: {best_epoch}')
#     print(res)