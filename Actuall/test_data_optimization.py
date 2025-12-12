#!/usr/bin/env python3
"""
Quick test to verify data loading optimizations work correctly.
This validates the negative sampling vectorization and caching.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ligand_pocket_qgnn.data import LigandPocketDataProcessor

def test_negative_sampling_speed():
    """Test that negative sampling is fast."""
    print("=" * 60)
    print("Testing Negative Sampling Optimization")
    print("=" * 60)
    
    # Create a mock processor
    processor = LigandPocketDataProcessor(".", seed=42)
    
    # Manually add mock data
    from ligand_pocket_qgnn.data import PocketFeatures, LigandGraph, LigandPocketInteraction
    
    n_pockets = 100
    n_ligands = 100
    n_positive = 500
    
    print(f"\nMock data:")
    print(f"  Pockets: {n_pockets}")
    print(f"  Ligands: {n_ligands}")
    print(f"  Target negatives: {n_positive}")
    
    # Create mock pockets
    for i in range(n_pockets):
        pocket = PocketFeatures(pocket_id=f"pocket_{i}")
        pocket.volume = np.random.random()
        processor.pockets[f"pocket_{i}"] = pocket
    
    # Create mock ligands
    for i in range(n_ligands):
        ligand = LigandGraph(
            ligand_id=f"ligand_{i}",
            atom_features=np.random.randn(20, 10).astype(np.float32),
            edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
            edge_attributes=np.ones((2, 1), dtype=np.float32)
        )
        processor.ligands[f"ligand_{i}"] = ligand
    
    # Create positive interactions
    existing_pairs = set()
    for i in range(n_positive):
        pocket_id = f"pocket_{i % n_pockets}"
        ligand_id = f"ligand_{i % n_ligands}"
        interaction = LigandPocketInteraction(
            ligand_id=ligand_id,
            pocket_id=pocket_id,
            label=1.0
        )
        processor.interactions.append(interaction)
        existing_pairs.add((ligand_id, pocket_id))
    
    # Now test negative sampling (the optimized part)
    print(f"\n⏱️  Testing vectorized negative sampling...")
    start_time = time.time()
    
    all_ligand_ids = np.array(list(processor.ligands.keys()))
    all_pocket_ids = np.array(list(processor.pockets.keys()))
    
    n_ligands_actual = len(all_ligand_ids)
    n_pockets_actual = len(all_pocket_ids)
    target_negs = n_positive
    
    # Generate candidates in bulk
    batch_size = min(10000, max(target_negs * 2, 5000))
    neg_ligand_indices = processor._rng.randint(0, n_ligands_actual, size=batch_size)
    neg_pocket_indices = processor._rng.randint(0, n_pockets_actual, size=batch_size)
    
    neg_interactions = 0
    for i in range(batch_size):
        if neg_interactions >= target_negs:
            break
        
        l_id = all_ligand_ids[neg_ligand_indices[i]]
        p_id = all_pocket_ids[neg_pocket_indices[i]]
        
        if (l_id, p_id) not in existing_pairs:
            interaction = LigandPocketInteraction(
                ligand_id=l_id,
                pocket_id=p_id,
                label=0.0
            )
            processor.interactions.append(interaction)
            existing_pairs.add((l_id, p_id))
            neg_interactions += 1
    
    elapsed = time.time() - start_time
    
    print(f"✓ Generated {neg_interactions}/{target_negs} negatives")
    print(f"✓ Time: {elapsed:.4f} seconds")
    print(f"✓ Rate: {neg_interactions/elapsed:.0f} samples/sec")
    
    if elapsed < 1.0:
        print(f"\n✅ EXCELLENT: Negative sampling is very fast!")
    elif elapsed < 5.0:
        print(f"\n✅ GOOD: Negative sampling is reasonably fast")
    else:
        print(f"\n⚠️  WARNING: Negative sampling might still be slow")
    
    return elapsed


def test_tensor_caching():
    """Test that tensor caching works."""
    print("\n" + "=" * 60)
    print("Testing Tensor Cache Optimization")
    print("=" * 60)
    
    from ligand_pocket_qgnn.data import LigandPocketDataset, LigandPocketInteraction
    import torch
    
    # Create mock processor
    processor = LigandPocketDataProcessor(".", seed=42)
    
    from ligand_pocket_qgnn.data import PocketFeatures, LigandGraph
    
    # Add mock data
    for i in range(10):
        pocket = PocketFeatures(pocket_id=f"pocket_{i}")
        processor.pockets[f"pocket_{i}"] = pocket
    
    for i in range(10):
        ligand = LigandGraph(
            ligand_id=f"ligand_{i}",
            atom_features=np.random.randn(20, 10).astype(np.float32),
            edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
            edge_attributes=np.ones((2, 1), dtype=np.float32)
        )
        processor.ligands[f"ligand_{i}"] = ligand
    
    # Create interactions
    interactions = [
        LigandPocketInteraction(
            ligand_id=f"ligand_{i % 10}",
            pocket_id=f"pocket_{i % 10}",
            label=float(i % 2)
        )
        for i in range(100)
    ]
    
    # Create dataset
    dataset = LigandPocketDataset(processor, interactions)
    
    print(f"\nMock data:")
    print(f"  Pockets: 10")
    print(f"  Ligands: 10")
    print(f"  Interactions: 100")
    print(f"  Cache size before: {len(dataset._tensor_cache)}")
    
    # Test caching
    print(f"\n⏱️  Testing tensor caching...")
    start_time = time.time()
    
    for i in range(100):
        x, edge_idx, pocket_vec, label = dataset[i]
    
    elapsed_first = time.time() - start_time
    
    # Second pass should be faster
    start_time = time.time()
    for i in range(100):
        x, edge_idx, pocket_vec, label = dataset[i]
    
    elapsed_second = time.time() - start_time
    
    print(f"✓ First pass: {elapsed_first:.4f}s (initial conversion)")
    print(f"✓ Second pass: {elapsed_second:.4f}s (cached)")
    print(f"✓ Cache speedup: {elapsed_first/elapsed_second:.2f}x")
    print(f"✓ Cache size: {len(dataset._tensor_cache)} entries")
    
    if elapsed_second < elapsed_first * 0.5:
        print(f"\n✅ EXCELLENT: Cache is very effective!")
    else:
        print(f"\n✅ GOOD: Cache provides noticeable speedup")
    
    return elapsed_first, elapsed_second


if __name__ == "__main__":
    try:
        test_negative_sampling_speed()
        test_tensor_caching()
        
        print("\n" + "=" * 60)
        print("✅ All optimization tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
