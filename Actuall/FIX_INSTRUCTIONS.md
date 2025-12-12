# Fix for DataLoader Error

## Problem
The notebook is using cached configuration with `num_workers=10`, but it should be `num_workers=0` for macOS/Jupyter.

## Solution

### Step 1: Restart Jupyter Kernel
In Jupyter, click: **Kernel** → **Restart Kernel**

### Step 2: Run This Test
Run this in a new cell to verify the fix:

```python
from hardware_optimizer import setup_environment
hw_info, config = setup_environment()
print(f"Workers: {config['num_workers']}")  # Should be 0
print(f"Pin Memory: {config['pin_memory']}")  # Should be False
```

You should see:
```
Workers: 0
Pin Memory: False
```

### Step 3: Re-run the Notebook
Now run all cells from the beginning.

## Why This Happened

The hardware_optimizer.py was updated to fix macOS/Jupyter compatibility, but your notebook kernel had the old version cached in memory.

## Alternative: Use a Python Script Instead

If you keep having issues with Jupyter, create a Python script instead:

```python
# train_comparison.py
from hardware_optimizer import setup_environment
from ligand_pocket_qgnn.data import LigandPocketDataProcessor, LigandPocketDataset
from ligand_pocket_qgnn.model import LigandPocketQGNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# ... rest of your code

if __name__ == '__main__':
    # Your training code here
    pass
```

Then run:
```bash
python train_comparison.py
```

This avoids Jupyter's multiprocessing issues entirely.

## Quick Fix: Override in Notebook

If you want to run right now without restarting, add this at the top of a cell:

```python
# Force reload the module
import importlib
import hardware_optimizer
importlib.reload(hardware_optimizer)

# Then get fresh config
from hardware_optimizer import setup_environment
hw_info, config = setup_environment()

# Override the dataloaders with correct settings
NUM_WORKERS = 0
PREFETCH = None
PIN_MEMORY = False

# Recreate loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,  # ← Fixed
    pin_memory=False,  # ← Fixed
    persistent_workers=False  # ← Must be False when num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,  # ← Fixed
    pin_memory=False,  # ← Fixed
    persistent_workers=False  # ← Must be False when num_workers=0
)
```

Then continue with training.
