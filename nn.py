import torch
from torch_geometric.nn import GCNConv, GATv2Conv
import torch.nn.functional as F

#this is the graph attention network module
class GraphAttentionNetwork(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, dim_in, dim_h, dim_out, heads=8):
    super().__init__()
    #we have 2 graph attention layers specifying dimension of inputs,outputs and heads
    self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
    self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
  def forward(self, data):
    #i get the input features and the graph connections as input x
    x, edge_index=data.x,data.edge_index
    #i filter the input features with a dropout layer avoid overfitting and improve generalization
    h = F.dropout(x, p=0.6, training=self.training)
    #i apply the first graph attention layer
    h = self.gat1(x, edge_index)
    #i apply a elu activation function as in the paper
    h = F.elu(h)
    #i filter the features h with a dropout layer avoid overfitting and improve generalization
    h = F.dropout(h, p=0.6, training=self.training)
    #i apply the second graph attention layer
    h = self.gat2(h, edge_index)
    #i return the logsoftmax of the output to normalize the output
    return h

#this is the graph convolutional network module
class GraphConvolutionalNetwork(torch.nn.Module):
    def __init__(self, in_size, h_size,out_size):
        super().__init__()
        #we have 2 graph convolutional layers specifying dimension of inputs,outputs and hiddend features
        self.conv1 = GCNConv(in_size, h_size)
        self.conv2 = GCNConv(h_size, out_size)

    def forward(self, data):
        #i get the input features and the graph connections as input x
        x, edge_index = data.x, data.edge_index
        #dropout layer to avoid overfitting and improve generalization
        x = F.dropout(x, p=0.6, training=self.training)
        #i apply the first graph convolutional layer
        x = self.conv1(x, edge_index)
        #i apply a relu activation function 
        x = F.relu(x)
         #i filter the features x with a dropout layer avoid overfitting and improve generalization
        x = F.dropout(x, p=0.6, training=self.training)
        #i apply the second graph convolutional layer
        x = self.conv2(x, edge_index)
        #i return the logsoftmax of the output to normalize the output
        return x


class Mlp(torch.nn.Module):
    def __init__(self, in_size, h_size,out_size):
        super().__init__()
        #we have 2 graph convolutional layers specifying dimension of inputs,outputs and hiddend features
        self.fc1 = torch.nn.Linear(in_size, h_size)
        self.fc2 = torch.nn.Linear(h_size, out_size)

    def forward(self, data):
        #i get the input features and the graph connections as input x
        x, edge_index = data.x, data.edge_index
        #dropout layer to avoid overfitting and improve generalization
        x = F.dropout(x, p=0.6, training=self.training)
        #i apply the first graph convolutional layer
        x = self.fc1(x)
        #i apply a relu activation function 
        x = F.relu(x)
         #i filter the features x with a dropout layer avoid overfitting and improve generalization
        x = F.dropout(x, p=0.6, training=self.training)
        #i apply the second graph convolutional layer
        x = self.fc2(x)
        #i return the logsoftmax of the output to normalize the output
        return x