import torch
from torch_geometric.nn import GCNConv, GATv2Conv
import torch.nn.functional as F



class GraphAttentionNetwork(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, dim_in, dim_h, dim_out, heads=8):
    super().__init__()
    self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
    self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)


  def forward(self, data):
    x, edge_index=data.x,data.edge_index
    h = F.dropout(x, p=0.6, training=self.training)
    h = self.gat1(x, edge_index)
    h = F.elu(h)
    h = F.dropout(h, p=0.6, training=self.training)
    h = self.gat2(h, edge_index)
    return F.log_softmax(h, dim=1)   

class GraphConvolutionalNetwork(torch.nn.Module):

    def __init__(self, in_size, h_size,out_size):
        super().__init__()
        self.conv1 = GCNConv(in_size, h_size)
        self.conv2 = GCNConv(h_size, out_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)