"""Getting system device information."""


import platform
import subprocess
import torch


def get_cpu_name():
    try:
        if platform.system() == 'Windows':
            cmd = ['wmic', 'cpu', 'get', 'name']
            output = subprocess.check_output(cmd, encoding='utf-8').strip()
            return output.split('\n')[1].strip()
        elif platform.system() == 'Darwin':  # macOS
            cmd = ['sysctl', '-n', 'machdep.cpu.brand_string']
            return subprocess.check_output(cmd, encoding='utf-8').strip()
        elif platform.system() == 'Linux':
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.strip() and line.rstrip('\n').startswith('model name'):
                        return line.rstrip('\n').split(':')[1].strip()
    except Exception:
        pass
    return 'Unknown_CPU'


def get_gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        return 'Unknown GPU'