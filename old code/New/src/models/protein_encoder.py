"""Protein encoder."""

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GATConv, global_mean_pool, global_add_pool


class ProteinContactGraphEncoder(nn.Module):
    """Protein encoder using contact graph."""
    
    def __init__(
        self,
        in_channels: int = 21,  # One-hot AA encoding
        hidden_channels: int = 256,
        out_channels: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        conv_type: str = 'gin'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.conv_type = conv_type
        
        # Initial linear
        self.lin_init = nn.Linear(in_channels, hidden_channels)
        
        # Convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        if conv_type == 'gin':
            for i in range(num_layers):
                nn_module = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                self.convs.append(GINConv(nn_module, train_eps=True))
        
        elif conv_type == 'gat':
            for i in range(num_layers):
                self.convs.append(
                    GATConv(
                        hidden_channels,
                        hidden_channels // 8,
                        heads=8,
                        dropout=dropout
                    )
                )
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Output
        self.lin_out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
    
    def forward(self, data):
        """
        Args:
            data: PyG Data object with x, edge_index, batch
        
        Returns:
            Embedding of shape (batch_size, out_channels)
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Initial embedding
        x = self.lin_init(x)
        x = torch.relu(x)
        
        # Message passing
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        if batch is not None:
            x = global_add_pool(x, batch)
        else:
            # Single graph: aggregate all nodes
            x = global_add_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        
        # Output
        x = self.lin_out(x)
        
        return x


class ProteinCNNEncoder(nn.Module):
    """Fast 1D CNN protein encoder from sequence."""
    
    def __init__(
        self,
        vocab_size: int = 21,
        embedding_dim: int = 128,
        hidden_channels: int = 256,
        out_channels: int = 256,
        dropout: float = 0.1,
        kernel_sizes: list = [3, 5, 7]
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=20)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, hidden_channels, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels * len(kernel_sizes), hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
    
    def forward(self, seq_ids, lengths=None):
        """
        Args:
            seq_ids: (batch_size, max_seq_len) tensor of AA indices
            lengths: (batch_size,) optional tensor of actual sequence lengths
        
        Returns:
            Embedding of shape (batch_size, out_channels)
        """
        x = self.embedding(seq_ids)  # (B, L, E)
        x = x.transpose(1, 2)  # (B, E, L)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x)  # (B, H, L)
            conv_out = torch.relu(conv_out)
            conv_out = self.pool(conv_out)  # (B, H, 1)
            conv_out = conv_out.squeeze(-1)  # (B, H)
            conv_outputs.append(conv_out)
        
        x = torch.cat(conv_outputs, dim=-1)  # (B, H*len(kernels))
        x = self.dropout(x)
        x = self.fc(x)  # (B, out_channels)
        
        return x
