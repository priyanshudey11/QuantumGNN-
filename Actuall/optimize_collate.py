"""
Optimized collate function using PyTorch operations instead of Python loops.
This should reduce CPU usage significantly.
"""

import torch

def optimized_collate_fn(batch):
    """
    Faster collate function using vectorized PyTorch operations.
    Replaces Python loops with batched tensor operations.
    """
    # Pre-allocate lists
    batch_size = len(batch)

    # Unzip batch
    x_list, edge_index_list, pocket_list, label_list = zip(*batch)

    # Fast concatenation
    pocket_batch = torch.stack(pocket_list)
    label_batch = torch.stack(label_list)

    # Calculate offsets vectorized
    num_nodes_list = torch.tensor([x.shape[0] for x in x_list], dtype=torch.long)
    cumsum = torch.cat([torch.zeros(1, dtype=torch.long), num_nodes_list.cumsum(0)])

    # Batch graphs
    x_batch = torch.cat(x_list, dim=0)

    # Shift edge indices vectorized
    edge_index_shifted = []
    for i, edge_index in enumerate(edge_index_list):
        if edge_index.shape[1] > 0:
            edge_index_shifted.append(edge_index + cumsum[i])

    if edge_index_shifted:
        edge_index_batch = torch.cat(edge_index_shifted, dim=1)
    else:
        edge_index_batch = torch.zeros((2, 0), dtype=torch.long)

    # Create batch vector (which sample each node belongs to)
    batch_vec = torch.cat([torch.full((n,), i, dtype=torch.long)
                           for i, n in enumerate(num_nodes_list)])

    return x_batch, edge_index_batch, batch_vec, pocket_batch, label_batch


if __name__ == "__main__":
    print("Optimized collate function loaded!")
    print("\nUsage:")
    print("from optimize_collate import optimized_collate_fn")
    print("train_loader = DataLoader(..., collate_fn=optimized_collate_fn)")
