import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc(x)
        return x

def train(model, loader, optimizer, criterion, num_epochs=40):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            output = model(data.x, data.edge_index)
            data.y = data.y.to(torch.float32)
            # print(output.dtype)
            # print(data.y.dtype)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        avg_loss = total_loss / len(loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}')

def test(model, loader):
    model.eval()
    total_loss = 0
    for data in loader:
        output = model(data.x, data.edge_index)
        loss = F.mse_loss(output, data.y)
        total_loss += loss.item() * data.num_graphs
    avg_loss = total_loss / len(loader.dataset)
    print(f'Test Mean Squared Error: {avg_loss}')