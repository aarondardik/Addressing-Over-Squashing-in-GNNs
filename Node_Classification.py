#Node Classification

import os
import torch
from torch.nn import Linear
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
from IPython.display import Javascript

from torch_geometric.utils import to_networkx

from ORC_FRC_Stats import orc_frc_correl, orc_frc_negative_correl, graph_mutual_info 

from Naive_Rewiring import * 





def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()







class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x




def train(model):
      optimizer = torch.optim.Adam(model.parameters())
      criterion = torch.nn.CrossEntropyLoss()
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test(model):
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc


def makeModel(toTrain=True, printLoss=False):
    if not toTrain:
        model = GCN(hidden_channels=16)
        model.eval()
        out = model(data.x, data.edge_index)
        visualize(out, color=data.y)
        return model, out 
    
    else:
        model = GCN(hidden_channels=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0175, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
    
        for epoch in range(1, 751):
            loss = train(model)
            if printLoss:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
                
        model.eval()
        out = model(data.x, data.edge_index)
        visualize(out, color=data.y)
        return model, out
        
        
        
        



if __name__ =="__main__":
    #Note the Planetoid dataset is lifted from the citation networks 'PubMed,' 'Cora,' 'CiteSeer.' Nodes represent documents and edges
    #represent citation links between the documents. Each node is represented by a 1433-dimensional bag-of-words feature vector.     
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    data = dataset[0]
    
    print()
    print(data)
    print('===========================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    
 
    m1, out1 = makeModel(False, True)
    test_acc = test(m1)
    print(f'Test Accuracy for untrained model: {test_acc:.4f}')
    
    m2, out2 = makeModel(True, True)
    test_acc = test(m2)
    print(f'Test Accuracy for trained model: {test_acc:.4f}')
    
    
    
    #THIS TAKES A VERY LONG TIME IF USING THE PLANETOID DATA SET. I HAVE COMMENTED IT OUT.
    #ONLY RUN IF YOU CHANGE THE DATASET TO EITHER 4-ARY, KARATE OR A SMALL COMPLETE GRAPH LIKE K5 (though complete graphs have uninteresting results)
    #modified_data = rewiring_algorithm_one(to_networkx(data), 5)
    
    #G = nx.karate_club_graph()
    #modified_data = rewiring_algorithm_one(G, 3)
    
    #G = nx.complete_graph(5)
    #modified_data = rewiring_algorithm_one(G, 3)
    
    #G = nx.full_rary_tree(4, 80)
    #modified_data = rewiring_algorithm_one(G, 3)
    
    #m3, out3 = makeModel(True, True)
    #test_acc = test(m3)
    #print(f'Test Accuracy for trained model: {test_acc:.4f}')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''
    model = GCN(hidden_channels=16)
    model.eval()
    out = model(data.x, data.edge_index)
    visualize(out, color=data.y)


    model = GCN(hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, 101):
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    

    test_acc = test()
    print(f'Test Accuracy: {test_acc:.4f}')

    model.eval()

    out = model(data.x, data.edge_index)
    visualize(out, color=data.y)
    '''
    
    
    