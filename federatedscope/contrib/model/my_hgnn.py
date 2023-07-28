import torch
import torch.nn.functional as F

from torch.nn import ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from federatedscope.register import register_model
import dhg
import time
import numpy as np
import copy
from copy import deepcopy
import torch
import torch.optim as optim
import torch.nn.functional as F

# class MyGCN(torch.nn.Module):
#     def __init__(self, 
#                 in_channels,
#                 out_channels,
#                 hidden=64,
#                 max_depth=2,
#                 dropout=.0):
#         super(MyGCN, self).__init__()
#         self.convs = ModuleList()
#         for i in range(max_depth):
#             if i == 0:
#                 self.convs.append(GCNConv(in_channels, hidden))
#             elif (i+1) == max_depth:
#                 self.convs.append(GCNConv(hidden, out_channels))
#             else:
#                 self.convs.append(GCNConv(hidden, hidden))
#         self.dropout = dropout

#     def froward(self, data):
#         if isinstance(data, Data):
#             x, edge_index = data.x, data.edge_index
#         elif isinstance(data, tuple):
#             x, edge_index = data
#         else:
#             raise TypeError('Unsupported data type!')
        
#         for i, conv in enumerate(self.convs):
#             x = conv(x, edge_index)
#             if (x+1) == len(self.convs):
#                 break
#             x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
#         return x

from dhg import Hypergraph
from dhg.data import Cooking200
from dhg.models import HGNN
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
import matplotlib.pyplot as plt


def train(net, X, A, lbls, train_idx, optimizer, epoch):
    net.train()
    st = time.time()
    optimizer.zero_grad()
    outs = net(X, A)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}')
    return loss.item()

@torch.no_grad()
def infer(net, X, A, lbls, idx, test=False):
    net.eval()
    outs = net(X, A)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(y_true=lbls, y_pred=outs)
    else:
        res = evaluator.test(lbls, outs)
    return res
    


if __name__ == '__main__':
    set_seed(2021)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    data = Cooking200()
    X, lbl = torch.eye(data["num_vertices"]), data["labels"]
    G = Hypergraph(data["num_vertices"], data["edge_list"])
    train_mask, val_mask, test_mask = data['train_mask'], data['val_mask'], data['test_mask']

    # print some basic information about the dataset
    print('some info. about the dataset: ............')
    print(X, X.shape)
    print(lbl, lbl.shape, torch.unique(lbl))
    print(G)
    print(train_mask, train_mask.shape)

    net = HGNN(X.shape[1], 32, data['num_classes'], use_bn=True)
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

    X, lbl = X.to(device), lbl.to(device)
    G = G.to(device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(200):
        # train
        train(net, X, G, lbl, train_mask, optimizer, epoch)
        # validation
        if epoch % 5 == 0:
            with torch.no_grad():
                val_res = infer(net, X, G, lbl, val_mask, test=False)
            if val_res > best_val:
                print(f'update best: {val_res:.5f}')
                best_val = val_res
                best_epoch, best_state = epoch, deepcopy(net.state_dict())
                torch.save(best_state,'hgnn_best_model_params.pt')
    print('\ntrain finished!')
    print(f"best val: {best_val:.5f}")
    # test
    print("test...")
    net.load_state_dict(best_state)
    res = infer(net, X, G, lbl, test_mask, test=True)
    print(f'final result: epoch: {best_epoch}')
    print(res)
    

def hgnnbuilder(model_config, input_shape):
    x_shape, num_label, num_edge_features = input_shape
    model = HGNN(X.shape[1], model_config.out_channels, hidden=model_config.hidden, dropout=model_config.dropout, use_bn=True)
    return model



def call_my_net(model_config, local_data):
    if model_config.type == 'gnn_myhgnn':
        model = hgnnbuilder(model_config, local_data)
        return model
    
register_model("gnn_myhgnn", call_my_net)
        
