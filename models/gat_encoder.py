import torch.nn as nn
from models.graph_convolutional_layer import GraphConvolution
from models.graph_attention_layer import GraphAttentionLayer


class GATEncoder(nn.Module):
    """
    Graph encoder using a Graph Attention Network.
    """
    def __init__(self, num_features, hidden_size, dropout=0):
        super(GATEncoder, self).__init__()

        self.gc1 = GraphConvolution(num_features, hidden_size)
        self.gc2 = GraphConvolution(hidden_size, hidden_size)

        self.attention1 = GraphAttentionLayer(num_features, hidden_size)
        self.attention2 = GraphAttentionLayer(hidden_size, hidden_size)

        self.dropout = dropout

    def forward(self, x, adj):
        x = self.attention1(x, adj)
        x = self.attention1(x, adj, concat=False)

        return x.mean(dim=0)

