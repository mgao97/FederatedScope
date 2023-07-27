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
    if config.data.type == "cooking200":
        data, modified_config = load_cooking200(config)
        return data, modified_config

register_data('cooking200', call_cooking200)


