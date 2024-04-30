import os 

import networkx as nx
import math 
import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ogb

from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

from ORC_FRC_Stats import orc_frc_correl, orc_frc_negative_correl, graph_mutual_info 

from Naive_Rewiring import naiveRewiring

from tqdm import tqdm

import torch 
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch_geometric
from torch_geometric.data import Data
#import pytorch_lightning
#import lightning 
#import torch_sparse

from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.datasets import TUDataset, Planetoid

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing, SAGEConv
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

import torch.nn.functional as F 
from torch_geometric.nn import GCNConv 








edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous())
print(data)
print("\n\nhello!!\n\n")
print(data.num_nodes)
dataset2 = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
dataset2 = dataset2.shuffle()
loader = DataLoader(dataset2, batch_size=32, shuffle=True)
print("Length of dataset is {} and number of classes is {}\n".format(len(dataset2), dataset2.num_classes))











#G = nx.karate_club_graph()

dataset = Planetoid(root='/tmp/Cora', name='Cora')
graph_v1 = dataset[0]
#print("\nPlanetoid\n")
#print(dataset.num_node_features)
#print(dataset[0].num_nodes)



list_of_dictionaries = []




class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    list_of_dictionaries.append((orc_frc_correl(G), orc_frc_negative_correl(G)))


model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')



#G = nx.caveman_graph(20, 7)
naiveRewiring(G)



print(len(list_of_dictionaries))




'''
G = nx.full_rary_tree(4, 80)
#G = nx.complete_graph(10)
#G = nx.random_internet_as_graph(5000)
d11, d12 = orc_frc_correl(G)
d21, d22 = orc_frc_negative_correl(G)
print(list(d11.keys()))
'''






    

