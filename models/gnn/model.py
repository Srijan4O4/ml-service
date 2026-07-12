"""model.py — Relational Graph Convolutional Network for seller-fraud detection."""
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCN(torch.nn.Module):
    """Two-layer RGCN over a heterogeneous (multi-relation) graph.

    in_feats   : input feature dimension per node (3 here)
    hidden     : hidden dimension
    num_rels   : number of edge/relation types (3: review, sold_by, bought)
    num_classes: output classes (2: legit vs fraud)
    """

    def __init__(self, in_feats, hidden, num_rels, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = RGCNConv(in_feats, hidden, num_relations=num_rels)
        self.conv2 = RGCNConv(hidden, hidden, num_relations=num_rels)
        self.lin = torch.nn.Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)

    def embed(self, x, edge_index, edge_type):
        """Return penultimate-layer node embeddings (for the fraud-network view)."""
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type))
        return x
