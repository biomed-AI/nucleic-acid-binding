import numpy as np
import torch
import torch.nn as nn
from gvp import GVP, GVPConvLayer, LayerNorm, tuple_index
from torch.distributions import Categorical
from torch_scatter import scatter_mean
from torch.nn import functional as F

class GVP_encoder(nn.Module):
    '''
    GVP-GNN for Model Quality Assessment as described in manuscript.
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]
    
    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param edge_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.1):
        
        super(GVP_encoder, self).__init__()
        activations = (F.relu, None)

        self.seq_in = seq_in
        if self.seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None), vector_gate=True)
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None), vector_gate=True)
        )
        
        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, activations=activations, vector_gate=True, drop_rate=drop_rate)
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0), activations=activations, vector_gate=True))
            
        self.dense = nn.Sequential(
            nn.Linear(ns, ns), nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            #nn.Linear(2*ns, 1)
        )

    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        if self.seq_in and seq is not None:
            seq = self.W_s(seq)
            h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)

        out = self.W_out(h_V)
        
        #if batch is None: out = out.mean(dim=0, keepdims=True)
        #else: out = scatter_mean(out, batch, dim=0)
        
        return self.dense(out) #self.dense(out).squeeze(-1) + 0.5


class GLMSite(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, augment_eps, dropout):
        super(GLMSite, self).__init__()

        self.GVP_encoder = GVP_encoder(node_in_dim=(input_dim,3), node_h_dim=(hidden_dim, 16), edge_in_dim=(32,1), edge_h_dim=(32, 1), seq_in=True, num_layers=num_layers, drop_rate=dropout)

        self.FC_DNA1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_DNA2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_RNA1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_RNA2 = nn.Linear(hidden_dim, 1, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, h_V, edge_index, h_E, seq):
  
        h_V = self.GVP_encoder(h_V, edge_index, h_E, seq) # [num_residue, hidden_dim]
        logits_DNA = self.FC_DNA2(F.elu(self.FC_DNA1(h_V))) # [num_residue, 1]
        logits_RNA = self.FC_RNA2(F.elu(self.FC_RNA1(h_V))) # [num_residue, 1]
        logits = torch.cat((logits_DNA, logits_RNA), 1) # [num_residue, 2]
        return logits
