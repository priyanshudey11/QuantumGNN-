import torch
import os
import multiprocessing as mp
import platform

def setup_environment():    
    # 1. Detect Device
    if torch.cuda.is_available():
        device = 'cuda'
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = 'mps'
        device_name = "Apple Silicon GPU (MPS)"
    else:
        device = 'cpu'
        device_name = "CPU"
        
    cpu_count = mp.cpu_count()
    
    # 2. Configure based on device
    if device == 'cuda':
        # GPU Optimization
        batch_size = 512
        num_workers = min(8, cpu_count)
        prefetch_factor = 2
        pin_memory = True
        quantum_device = 'lightning.gpu' # PennyLane-Lightning-GPU
        
    elif device == 'mps':
        # Apple Silicon Optimization
        batch_size = 256
        num_workers = max(1, cpu_count - 1)
        prefetch_factor = 2
        pin_memory = True
        quantum_device = 'default.qubit' # MPS doesn't have a stable PennyLane plugin yet usually
        
    else: # CPU
        # CPU Optimization
        batch_size = 128
        num_workers = max(1, cpu_count - 1)
        prefetch_factor = 2
        pin_memory = False
        quantum_device = 'lightning.qubit' # Faster CPU simulator
        
    hw_info = {
        'device': device,
        'device_name': device_name,
        'cpu_count': cpu_count
    }
    
    hw_config = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'prefetch_factor': prefetch_factor,
        'pin_memory': pin_memory,
        'quantum_device': quantum_device
    }
    
    print(f"Hardware Optimization: {device_name}")
    print(f"Config: Batch={batch_size}, Workers={num_workers}, QDevice={quantum_device}")
    
    return hw_info, hw_config
