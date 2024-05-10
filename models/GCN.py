import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score
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
        x = torch.mean(x, dim=0, keepdim=True)
        x = F.dropout(x, training=self.training)
        x = self.fc(x)
        return torch.sigmoid(x) 

def train(model, loader, optimizer, criterion, num_epochs=40):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            output = model(data.x, data.edge_index)
            output = torch.round(output)
            data.y = data.y.to(torch.float32)
            # print(output.dtype)
            # print(data.y.dtype)
            # import pdb; pdb.set_trace()
            loss = criterion(output[0], data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        avg_loss = total_loss / len(loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}')

def test(model, loader):
    model.eval()
    total_loss = 0
    total_abs_error = 0
    total_true_value = 0
    
    for data in loader:
        output = model(data.x, data.edge_index)
        loss = F.l1_loss(output, data.y)  # Using L1 loss for absolute error
        total_loss += loss.item() * data.num_graphs
        total_abs_error += torch.sum(torch.abs(output - data.y)).item()  # Sum of absolute errors
        total_true_value += torch.sum(data.y).item()  # Sum of true values
    
    avg_loss = total_loss / len(loader.dataset)
    avg_abs_error = total_abs_error / len(loader.dataset)
    avg_true_value = total_true_value / len(loader.dataset)
    
    # Compute MAPE
    mape = (total_abs_error / total_true_value) * 100 if total_true_value != 0 else 0
    
    print(f'Test Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
    print(f'Test Mean Absolute Error (MAE): {avg_abs_error:.2f}')
    print(f'Test Mean Squared Error (MSE): {avg_loss:.2f}')