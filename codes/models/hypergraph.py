import math
import torch
from torch import nn


class HypergraphLearning(nn.Module):
    def __init__(self, hidden_dim, num_nodes, num_edges):
        super(HypergraphLearning, self).__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.edge_clf = torch.randn(hidden_dim, self.num_edges) / math.sqrt(self.num_edges)
        self.edge_clf = nn.Parameter(self.edge_clf, requires_grad=True)
        self.edge_map = torch.randn(self.num_edges, self.num_edges) / math.sqrt(self.num_edges)
        self.edge_map = nn.Parameter(self.edge_map, requires_grad=True)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pre_h_adj=None):
        # feat = x.reshape(x.size(0), -1, x.size(3))
        feat = x.reshape(-1, self.num_nodes, x.size(-1))
        hyper_assignment = torch.softmax(feat @ self.edge_clf, dim=-1)
        # if pre_h_adj is None:
        #     pre_h_adj = torch.zeros((hyper_assignment.shape[0], hyper_assignment.shape[1], 12), dtype=hyper_assignment.dtype, device=hyper_assignment.device)
        # hyper_assignment = torch.cat([hyper_assignment, pre_h_adj], dim=-1)
        hyper_feat = hyper_assignment.transpose(1, 2) @ feat
        hyper_feat_mapped = self.activation(self.edge_map @ hyper_feat)
        hyper_out = hyper_feat_mapped + hyper_feat
        y = self.activation(hyper_assignment @ hyper_out)
        # y = y.reshape(x.size(0), x.size(1), x.size(2), x.size(3))
        y = y.reshape(x.shape)
        y_final = self.norm(y + x)
        return y_final
    