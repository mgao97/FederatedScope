# Import FederatedScope modules
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, get_server_cls
from federatedscope.core.configs.config import global_cfg, CfgNode
from federatedscope.core.auxiliaries.runner_builder import get_runner

# Import your custom GNN model and dataset functions
from federatedscope.contrib.model.my_gcn import MyGCN
from federatedscope.contrib.data.load_hypergraph_clique_Cooking200 import Cooking200

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


def load_clique_cooking200(config):

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
        data, modified_config = load_clique_cooking200(config)
        return data, modified_config

register_data('cooking200', call_cooking200)

if __name__ == '__main__':
    # Initialize the configuration with global_cfg.clone()
    init_cfg = global_cfg.clone()

    # Parse command-line arguments
    args = parse_args()

    # Merge configurations from the configuration file and command-line options
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    # Update logger and set random seed
    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # Load clients' cfg file
    if args.client_cfg_file:
        client_cfgs = CfgNode.load_cfg(open(args.client_cfg_file, 'r'))
        client_cfgs.merge_from_list(client_cfg_opt)
    else:
        client_cfgs = None

    # Load your custom dataset using load_custom_dataset() function
    data, modified_cfg = call_cooking200(config=init_cfg.clone(), client_cfgs=client_cfgs)
    init_cfg.merge_from_other_cfg(modified_cfg)

    init_cfg.freeze()

    # Create the federated GNN model using your custom model class MyGCN
    gnn_model = MyGCN(data.x_shape[-1],
                20,
                32,
                2,
                .0)

    # Create the federated runner
    runner = get_runner(data=data,
                        server_class=get_server_cls(init_cfg),
                        client_class=get_client_cls(init_cfg),
                        config=init_cfg.clone(),
                        client_configs=client_cfgs,
                        model=gnn_model)

    # Train the GNN model in a federated learning manner
    _ = runner.run()
