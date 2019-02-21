import torch.nn as nn
import torch.nn.functional as F
from models.graph_convolution import GraphConvolution


class GCNEncoder(nn.Module):
    def __init__(self, num_features, hidden_size, dropout=0):
        super(GCNEncoder, self).__init__()

        self.gc1 = GraphConvolution(num_features, hidden_size)
        self.gc2 = GraphConvolution(hidden_size, hidden_size)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return x.mean(dim=0)

