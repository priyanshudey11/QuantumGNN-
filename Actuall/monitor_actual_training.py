"""
Monitor CPU usage during ACTUAL training to see real-world performance
"""

import psutil
import time
import sys
from datetime import datetime

def find_python_processes():
    """Find all Python processes"""
    procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent']):
        try:
            if 'python' in proc.info['name'].lower():
                procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return procs

print(f"{'='*80}")
print(f"🔍 MONITORING ACTUAL TRAINING CPU USAGE")
print(f"{'='*80}\n")

# Find Python processes
python_procs = find_python_processes()

if not python_procs:
    print("⚠️  No Python processes found!")
    print("Make sure your training is running!")
    sys.exit(1)

print(f"Found {len(python_procs)} Python process(es):")
for proc in python_procs:
    try:
        cmdline = ' '.join(proc.cmdline()[:4])
        print(f"  PID {proc.pid}: {cmdline}")
    except:
        print(f"  PID {proc.pid}: (cannot read cmdline)")

print(f"\n{'='*80}")
print(f"LIVE MONITORING (30 seconds)")
print(f"{'='*80}\n")

cpu_count = psutil.cpu_count()
print(f"System: {cpu_count} CPU cores\n")

# Headers
print(f"{'Time':<10} {'Total':<8} {'Python':<8} {'Per-Core (first 12)'}")
print(f"{'-'*80}")

samples = []
start_time = time.time()
duration = 30

try:
    while time.time() - start_time < duration:
        # Get system CPU
        total_cpu = psutil.cpu_percent(interval=0.1)
        per_core = psutil.cpu_percent(interval=0, percpu=True)

        # Get Python CPU
        python_cpu = 0
        for proc in python_procs:
            try:
                python_cpu += proc.cpu_percent(interval=0)
            except:
                pass

        # Format output
        timestamp = datetime.now().strftime('%H:%M:%S')
        core_str = ' '.join([f"{int(c):2d}" for c in per_core[:12]])

        print(f"{timestamp:<10} {total_cpu:5.1f}%  {python_cpu:5.1f}%  {core_str}")

        # Store
        samples.append({
            'total': total_cpu,
            'python': python_cpu,
            'per_core': per_core
        })

        time.sleep(0.5)

except KeyboardInterrupt:
    print(f"\n\nStopped by user")

# Analysis
if samples:
    print(f"\n{'='*80}")
    print(f"RESULTS ({len(samples)} samples over {time.time() - start_time:.1f}s)")
    print(f"{'='*80}\n")

    avg_total = sum(s['total'] for s in samples) / len(samples)
    avg_python = sum(s['python'] for s in samples) / len(samples)
    max_total = max(s['total'] for s in samples)
    max_python = max(s['python'] for s in samples)

    print(f"Overall System CPU:")
    print(f"  Average: {avg_total:.1f}%")
    print(f"  Maximum: {max_total:.1f}%")

    print(f"\nPython Process CPU:")
    print(f"  Average: {avg_python:.1f}%")
    print(f"  Maximum: {max_python:.1f}%")

    # Per-core
    import numpy as np
    avg_per_core = np.mean([s['per_core'] for s in samples], axis=0)

    print(f"\nPer-Core Average:")
    for i in range(min(cpu_count, 12)):
        bar = '█' * int(avg_per_core[i] / 5)
        print(f"  Core {i:2d}: {avg_per_core[i]:5.1f}%  {bar}")

    # Verdict
    print(f"\n{'='*80}")
    print(f"VERDICT")
    print(f"{'='*80}\n")

    if avg_python < 100:
        print(f"⚠️  Python using {avg_python:.1f}% CPU (low)")
        print(f"   This is EXPECTED for sequential quantum simulation")
        print(f"   Only 1-2 cores active at a time")
    elif avg_python < 400:
        print(f"✓ Python using {avg_python:.1f}% CPU (moderate)")
        print(f"  Equivalent to {avg_python/100:.1f} cores fully utilized")
        print(f"  This is GOOD for quantum simulation!")
    elif avg_python < 800:
        print(f"✓✓ Python using {avg_python:.1f}% CPU (good)")
        print(f"  Equivalent to {avg_python/100:.1f} cores fully utilized")
        print(f"  Excellent for quantum simulation!")
    else:
        print(f"✓✓✓ Python using {avg_python:.1f}% CPU (excellent)")
        print(f"  Equivalent to {avg_python/100:.1f} cores fully utilized")
        print(f"  Maximum efficiency!")

    # Check if it's actually training
    core_variance = np.std(avg_per_core[:12])
    if avg_python < 50:
        print(f"\n⚠️  Very low CPU usage - is training actually running?")
        print(f"   Make sure training loop is executing")
    elif core_variance < 5:
        print(f"\n✓ Even core distribution - good parallelization")
    else:
        print(f"\n⚠️  Uneven core distribution (expected for quantum)")

print(f"\n{'='*80}\n")
