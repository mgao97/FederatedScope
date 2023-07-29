import torch
import torch.nn.functional as F

from torch.nn import ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from federatedscope.register import register_model


class MyGCN(torch.nn.Module):
    def __init__(self, 
                in_channels,
                out_channels,
                hidden=64,
                max_depth=2,
                dropout=.0):
        super(MyGCN, self).__init__()
        self.convs = ModuleList()
        for i in range(max_depth):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden))
            elif (i+1) == max_depth:
                self.convs.append(GCNConv(hidden, out_channels))
            else:
                self.convs.append(GCNConv(hidden, hidden))
        self.dropout = dropout

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if (x+1) == len(self.convs):
                break
            x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x
    
def gcnbuilder(model_config, input_shape):
    x_shape, num_label, num_edge_features = input_shape
    model = MyGCN(x_shape[-1],
                model_config.out_channels,
                hidden=model_config.hidden,
                max_depth=model_config.layer,
                dropout=model_config.dropout)
    return model

def call_my_net(model_config, local_data):
    if model_config.type == 'gnn_mygcn':
        model = gcnbuilder(model_config, local_data)
        return model
    
register_model("gnn_mygcn", call_my_net)
        