"""Drug GNN encoder."""

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GATConv, global_mean_pool, global_add_pool


class DrugGNN(nn.Module):
    """Graph neural network for drug encoding."""
    
    def __init__(
        self,
        in_channels: int = 31,  # Node feature dimension
        edge_in_channels: int = 6,  # Edge feature dimension
        hidden_channels: int = 256,
        out_channels: int = 256,
        num_layers: int = 4,
        conv_type: str = 'gin',  # 'gin' or 'gat'
        dropout: float = 0.1,
        add_edge_features: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.edge_in_channels = edge_in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.conv_type = conv_type
        self.add_edge_features = add_edge_features
        self.dropout = nn.Dropout(dropout)
        
        # Initial linear layer
        self.lin_init = nn.Linear(in_channels, hidden_channels)
        
        # Convolution layers
        self.convs = nn.ModuleList()
        
        if conv_type == 'gin':
            for i in range(num_layers):
                if i == 0:
                    nn_module = nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.BatchNorm1d(hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, hidden_channels),
                    )
                else:
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
                        dropout=dropout,
                        edge_dim=edge_in_channels if add_edge_features else None
                    )
                )
        
        # Output layer
        self.lin_out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])
    
    def forward(self, data):
        """
        Args:
            data: PyG Data object with x, edge_index, edge_attr (optional), batch
        
        Returns:
            Embedding of shape (batch_size, out_channels)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if self.add_edge_features else None
        batch = data.batch
        
        # Initial embedding
        x = self.lin_init(x)
        x = torch.relu(x)
        
        # Message passing
        for i, conv in enumerate(self.convs):
            if self.conv_type == 'gin':
                x = conv(x, edge_index)
            else:  # gat
                x = conv(x, edge_index, edge_attr=edge_attr)
            
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        x = global_add_pool(x, batch)
        
        # Output projection
        x = self.lin_out(x)
        
        return x
