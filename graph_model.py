import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, global_mean_pool, global_max_pool

class GCN_MIL_MultiClass(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = SAGPooling(hidden_channels, ratio=0.5)

        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.pool2 = SAGPooling(hidden_channels, ratio=0.5)

        self.gcn3 = GCNConv(hidden_channels, hidden_channels)

        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_classes)

        self.dropout = nn.Dropout(0.2)
        self.dropout_fc2 = nn.Dropout(0.1) 


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.leaky_relu(self.gcn1(x, edge_index), negative_slope=0.01)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)

        x = F.leaky_relu(self.gcn2(x, edge_index), negative_slope=0.01)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)

        x = F.leaky_relu(self.gcn3(x, edge_index), negative_slope=0.01)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        x = self.dropout(x)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout_fc2(x)
        
        return self.fc2(x)  # logits only, no softmax


    def get_embedding(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.leaky_relu(self.gcn1(x, edge_index), negative_slope=0.01)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)

        x = F.leaky_relu(self.gcn2(x, edge_index), negative_slope=0.01)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)

        x = F.leaky_relu(self.gcn3(x, edge_index), negative_slope=0.01)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        x = self.dropout(x)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout(x)

        return x




def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, GCNConv):
        nn.init.xavier_uniform_(m.lin.weight)
        if m.lin.bias is not None:
            nn.init.zeros_(m.lin.bias)

