import os
import platform
import subprocess
import torch
import multiprocessing as mp


def detect_hardware():
    # Automatically detect and characterize hardware configuration.

    hw_info = {
        'platform': platform.system(),
        'cpu_count': mp.cpu_count(),
        'cpu_brand': 'Unknown',
        'cpu_vendor': 'unknown',
        'device': 'cpu',
        'device_name': 'CPU',
        'memory_gb': 0,
        'compute_capability': 'generic'
    }

    # CPU Detection
    try:
        if platform.system() == 'Darwin':  # macOS
            brand = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                           text=True, timeout=2).strip()
            hw_info['cpu_brand'] = brand

            mem_bytes = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'],
                                                    text=True, timeout=2).strip())
            hw_info['memory_gb'] = mem_bytes / (1024**3)

            # Detect Apple Silicon
            if 'Apple' in brand:
                hw_info['cpu_vendor'] = 'apple'
                if 'M4' in brand:
                    hw_info['compute_capability'] = 'apple_m4'
                elif 'M3' in brand:
                    hw_info['compute_capability'] = 'apple_m3'
                elif 'M2' in brand:
                    hw_info['compute_capability'] = 'apple_m2'
                elif 'M1' in brand:
                    hw_info['compute_capability'] = 'apple_m1'
                else:
                    hw_info['compute_capability'] = 'apple_silicon'
            elif 'Intel' in brand:
                hw_info['cpu_vendor'] = 'intel'
                hw_info['compute_capability'] = 'intel_mac'

        elif platform.system() == 'Linux':
            # Read CPU info
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'model name' in cpuinfo:
                    for line in cpuinfo.split('\n'):
                        if 'model name' in line:
                            brand = line.split(':')[1].strip()
                            hw_info['cpu_brand'] = brand
                            break

                    # Detect CPU vendor
                    brand_lower = brand.lower()
                    if 'amd' in brand_lower:
                        hw_info['cpu_vendor'] = 'amd'
                        if 'epyc' in brand_lower:
                            hw_info['compute_capability'] = 'amd_epyc'
                        elif 'threadripper' in brand_lower:
                            hw_info['compute_capability'] = 'amd_threadripper'
                        elif 'ryzen' in brand_lower:
                            if '9' in brand:
                                hw_info['compute_capability'] = 'amd_ryzen9'
                            elif '7' in brand:
                                hw_info['compute_capability'] = 'amd_ryzen7'
                            else:
                                hw_info['compute_capability'] = 'amd_ryzen'
                        else:
                            hw_info['compute_capability'] = 'amd'

                    elif 'intel' in brand_lower:
                        hw_info['cpu_vendor'] = 'intel'
                        if 'xeon' in brand_lower:
                            hw_info['compute_capability'] = 'intel_xeon'
                        elif 'i9' in brand_lower:
                            hw_info['compute_capability'] = 'intel_i9'
                        elif 'i7' in brand_lower:
                            hw_info['compute_capability'] = 'intel_i7'
                        elif 'i5' in brand_lower:
                            hw_info['compute_capability'] = 'intel_i5'
                        else:
                            hw_info['compute_capability'] = 'intel'

                    elif any(x in brand_lower for x in ['snapdragon', 'qualcomm']):
                        hw_info['cpu_vendor'] = 'qualcomm'
                        hw_info['compute_capability'] = 'snapdragon'

                    elif 'arm' in brand_lower or 'neoverse' in brand_lower:
                        hw_info['cpu_vendor'] = 'arm'
                        hw_info['compute_capability'] = 'arm_server'

            # Read memory info
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        mem_kb = int(line.split()[1])
                        hw_info['memory_gb'] = mem_kb / (1024**2)
                        break

        elif platform.system() == 'Windows':
            try:
                import wmi
                c = wmi.WMI()

                # Get CPU info
                for cpu in c.Win32_Processor():
                    hw_info['cpu_brand'] = cpu.Name
                    brand_lower = cpu.Name.lower()

                    if 'amd' in brand_lower:
                        hw_info['cpu_vendor'] = 'amd'
                        if 'threadripper' in brand_lower:
                            hw_info['compute_capability'] = 'amd_threadripper'
                        elif 'ryzen' in brand_lower:
                            hw_info['compute_capability'] = 'amd_ryzen'
                        else:
                            hw_info['compute_capability'] = 'amd'
                    elif 'intel' in brand_lower:
                        hw_info['cpu_vendor'] = 'intel'
                        if 'xeon' in brand_lower:
                            hw_info['compute_capability'] = 'intel_xeon'
                        else:
                            hw_info['compute_capability'] = 'intel'
                    elif any(x in brand_lower for x in ['snapdragon', 'qualcomm']):
                        hw_info['cpu_vendor'] = 'qualcomm'
                        hw_info['compute_capability'] = 'snapdragon'
                    break

                # Get memory
                for mem in c.Win32_ComputerSystem():
                    hw_info['memory_gb'] = int(mem.TotalPhysicalMemory) / (1024**3)
                    break
            except ImportError:
                # Fallback if wmi not available
                pass

    except Exception as e:
        print(f"Warning: Could not fully detect CPU: {e}")

    # GPU/Accelerator Detection
    if torch.cuda.is_available():
        hw_info['device'] = 'cuda'
        hw_info['device_name'] = torch.cuda.get_device_name(0)
        hw_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Detect NVIDIA architecture
        capability = torch.cuda.get_device_capability(0)
        compute_ver = capability[0] * 10 + capability[1]

        device_name_lower = hw_info['device_name'].lower()

        # Classify by generation/tier
        if compute_ver >= 90:  # Hopper and beyond
            hw_info['compute_capability'] = 'nvidia_hopper_h100'
        elif compute_ver >= 80:  # Ampere
            if 'a100' in device_name_lower or 'a800' in device_name_lower:
                hw_info['compute_capability'] = 'nvidia_ampere_datacenter'
            elif any(x in device_name_lower for x in ['rtx 40', 'rtx 4090', 'rtx 4080']):
                hw_info['compute_capability'] = 'nvidia_ada_high'
            elif 'rtx 40' in device_name_lower:
                hw_info['compute_capability'] = 'nvidia_ada_mid'
            else:
                hw_info['compute_capability'] = 'nvidia_ampere'
        elif compute_ver >= 75:  # Turing
            hw_info['compute_capability'] = 'nvidia_turing'
        else:
            hw_info['compute_capability'] = f'nvidia_sm{compute_ver}'

        torch.backends.cudnn.benchmark = True

    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        hw_info['device'] = 'mps'
        hw_info['device_name'] = 'Apple Metal Performance Shaders'
        # Inherit compute capability from CPU detection
        if hw_info['compute_capability'].startswith('apple_'):
            hw_info['compute_capability'] = hw_info['compute_capability'] + '_gpu'
        else:
            hw_info['compute_capability'] = 'apple_silicon_gpu'

    return hw_info


def optimize_for_hardware(hw_info):
    # Determine optimal training configuration based on hardware.
    # Returns optimized batch size, workers, prefetch, etc.
    device = hw_info['device']
    capability = hw_info['compute_capability']
    mem_gb = hw_info['memory_gb']
    cores = hw_info['cpu_count']

    config = {
        'batch_size': 512,
        'num_workers': 4,
        'prefetch_factor': 2,
        'pin_memory': False,
        'quantum_device': 'default.qubit',
        'persistent_workers': True
    }

    # NVIDIA CUDA CONFIGURATIONS
    if device == 'cuda':
        config['pin_memory'] = True
        config['quantum_device'] = 'default.qubit'

        # High-end datacenter GPUs (A100, H100)
        if 'datacenter' in capability or 'hopper' in capability:
            config['batch_size'] = 4096
            config['num_workers'] = 12
            config['prefetch_factor'] = 6

        # High-end consumer (RTX 4090, 4080, 3090)
        elif mem_gb >= 20:
            config['batch_size'] = 2048
            config['num_workers'] = 8
            config['prefetch_factor'] = 4

        # Upper mid-range (RTX 4070 Ti, 3080)
        elif mem_gb >= 12:
            config['batch_size'] = 1024
            config['num_workers'] = 6
            config['prefetch_factor'] = 3

        # Mid-range (RTX 4060 Ti, 3070)
        elif mem_gb >= 8:
            config['batch_size'] = 512
            config['num_workers'] = 4
            config['prefetch_factor'] = 2

        # Entry-level (RTX 3060, older GPUs)
        else:
            config['batch_size'] = 256
            config['num_workers'] = 4
            config['prefetch_factor'] = 2

    # APPLE METAL (MPS) CONFIGURATIONS
    elif device == 'mps':
        config['pin_memory'] = True

        # Apple Silicon has unified memory architecture
        if 'm4' in capability:
            # M4: Latest, best performance
            config['batch_size'] = 1536
            config['num_workers'] = min(10, cores - 2)
            config['prefetch_factor'] = 5

        elif 'm3' in capability:
            # M3: High performance
            config['batch_size'] = 1024
            config['num_workers'] = min(8, cores - 2)
            config['prefetch_factor'] = 4

        elif 'm2' in capability:
            # M2: Good performance
            config['batch_size'] = 768
            config['num_workers'] = min(6, cores - 2)
            config['prefetch_factor'] = 3

        elif 'm1' in capability:
            # M1: Still capable
            config['batch_size'] = 512
            config['num_workers'] = 4
            config['prefetch_factor'] = 2

        else:
            # Generic Apple Silicon
            config['batch_size'] = 512
            config['num_workers'] = 4
            config['prefetch_factor'] = 2

    # CPU-ONLY CONFIGURATIONS
    else:
        # Apple Silicon CPU mode (no GPU)
        if 'apple' in capability:
            config['batch_size'] = 1024
            config['num_workers'] = max(6, cores // 2)
            config['prefetch_factor'] = 4

        # AMD Configurations
        elif 'amd' in capability:
            if 'epyc' in capability or 'threadripper' in capability:
                # High-end AMD (EPYC, Threadripper)
                config['batch_size'] = 3072
                config['num_workers'] = min(16, cores - 4)
                config['prefetch_factor'] = 8

            elif 'ryzen9' in capability or cores >= 16:
                # Ryzen 9, high core count
                config['batch_size'] = 2048
                config['num_workers'] = min(12, cores - 2)
                config['prefetch_factor'] = 6

            elif 'ryzen7' in capability or cores >= 8:
                # Ryzen 7, mid-high cores
                config['batch_size'] = 1024
                config['num_workers'] = min(8, cores - 2)
                config['prefetch_factor'] = 4

            else:
                # Ryzen 5 or lower
                config['batch_size'] = 512
                config['num_workers'] = max(4, cores // 2)
                config['prefetch_factor'] = 3

        # Intel Configurations
        elif 'intel' in capability:
            if 'xeon' in capability:
                # Intel Xeon (server)
                config['batch_size'] = 2048
                config['num_workers'] = min(12, cores - 4)
                config['prefetch_factor'] = 6

            elif 'i9' in capability or cores >= 16:
                # Intel i9
                config['batch_size'] = 1536
                config['num_workers'] = min(10, cores - 2)
                config['prefetch_factor'] = 4

            elif 'i7' in capability or cores >= 8:
                # Intel i7
                config['batch_size'] = 1024
                config['num_workers'] = min(6, cores - 2)
                config['prefetch_factor'] = 3

            else:
                # Intel i5 or lower
                config['batch_size'] = 512
                config['num_workers'] = max(4, cores // 2)
                config['prefetch_factor'] = 2

        # Qualcomm Snapdragon (ARM Windows/Linux)
        elif 'snapdragon' in capability:
            # Snapdragon X Elite and similar
            config['batch_size'] = 768
            config['num_workers'] = min(8, cores // 2)
            config['prefetch_factor'] = 3

        # Generic ARM (Neoverse, etc.)
        elif 'arm' in capability:
            if cores >= 64:  # ARM server (Graviton, Neoverse)
                config['batch_size'] = 2048
                config['num_workers'] = min(16, cores // 4)
                config['prefetch_factor'] = 6
            else:
                config['batch_size'] = 768
                config['num_workers'] = min(8, cores // 2)
                config['prefetch_factor'] = 3

        # Generic/Unknown CPU
        else:
            config['batch_size'] = 512
            config['num_workers'] = max(2, cores // 3)
            config['prefetch_factor'] = 2

    # Memory-based adjustments (prevent OOM)
    if mem_gb > 0:
        if mem_gb < 8:
            # Low memory: reduce batch size
            config['batch_size'] = min(config['batch_size'], 256)
        elif mem_gb < 16:
            config['batch_size'] = min(config['batch_size'], 512)
        elif mem_gb < 32:
            config['batch_size'] = min(config['batch_size'], 1024)

    return config


def print_hardware_info(hw_info, config):
    # Pretty print detected hardware and configuration.
    print(f"HARDWARE DETECTED")
    print(f"Platform:     {hw_info['platform']}")
    print(f"CPU:          {hw_info['cpu_brand']}")
    print(f"CPU Vendor:   {hw_info['cpu_vendor'].upper()}")
    print(f"CPU Cores:    {hw_info['cpu_count']}")
    print(f"Device:       {hw_info['device_name']}")
    print(f"Memory:       {hw_info['memory_gb']:.1f} GB")
    print(f"Capability:   {hw_info['compute_capability']}")
    print(f"AUTO-OPTIMIZED CONFIGURATION")
    print(f"Batch Size:   {config['batch_size']}")
    print(f"Workers:      {config['num_workers']}")
    print(f"Prefetch:     {config['prefetch_factor']}")
    print(f"Pin Memory:   {config['pin_memory']}")
    print(f"Quantum Dev:  {config['quantum_device']}")
    print(f"{'='*70}\n")


def setup_environment():
    hw_info = detect_hardware()
    config = optimize_for_hardware(hw_info)
    print_hardware_info(hw_info, config)
    return hw_info, config


if __name__ == '__main__':
    # Test the hardware detection
    hw_info, config = setup_environment()
