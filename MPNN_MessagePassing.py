#MPNN File for Message Passing GNN

import torch
import os 
import torch_scatter
import torch_sparse
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import ClusterData, ClusterLoader

dataset = Planetoid(root='data/Planetoid', name='PubMed',
                    transform=NormalizeFeatures())

print()
print(f'Dataset: {dataset}:')
print('==================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)




cluster_data = ClusterData(data, num_parts=128)  # 1. Create subgraphs.
train_loader = ClusterLoader(cluster_data, batch_size=32,
                             shuffle=True)  # 2. Stochastic partioning scheme.



criterion = torch.nn.CrossEntropyLoss()


def train(model):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=5e-4)
    for sub_data in train_loader:  # Iterate over each mini-batch.
        optimizer.zero_grad()  # Clear gradients.
        out = model(sub_data.x,
                    sub_data.edge_index)  # Perform a single forward pass.
        loss = criterion(
            out[sub_data.train_mask], sub_data.y[sub_data.train_mask]
        )  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.


def test(model):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[
            mask]  # Check against ground-truth labels.
        accs.append(int(correct.sum()) /
                    int(mask.sum()))  # Derive ratio of correct predictions.
    return accs


def run(model, epochs=5):
    for epoch in range(epochs):
        loss = train(model)
        train_acc, val_acc, test_acc = test(model)
        print(
            f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}'
        )



import copy

import torch.nn.functional as F
from torch_geometric.nn import (
    Aggregation,
    MaxAggregation,
    MeanAggregation,
    MultiAggregation,
    SAGEConv,
    SoftmaxAggregation,
    StdAggregation,
    SumAggregation,
    VarAggregation,
)


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, aggr='mean', aggr_kwargs=None):
        super().__init__()
        self.conv1 = SAGEConv(
            dataset.num_node_features,
            hidden_channels,
            aggr=aggr,
            aggr_kwargs=aggr_kwargs,
        )
        self.conv2 = SAGEConv(
            hidden_channels,
            dataset.num_classes,
            aggr=copy.deepcopy(aggr),
            aggr_kwargs=aggr_kwargs,
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x



seed = 42
torch.manual_seed(seed)
model = GNN(16, aggr='mean')
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
run(model)

















































'''
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
'''