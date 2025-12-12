"""
Live CPU monitoring DURING quantum training to see what's actually happening
Run this alongside your training notebook
"""

import psutil
import time
import os
import sys
from datetime import datetime

def monitor_cpu_live(duration=60, interval=0.5):
    """
    Monitor CPU usage in real-time

    Args:
        duration: How long to monitor (seconds)
        interval: How often to sample (seconds)
    """

    print(f"{'='*70}")
    print(f"🔍 LIVE CPU MONITORING")
    print(f"{'='*70}")
    print(f"Duration: {duration}s")
    print(f"Sampling interval: {interval}s")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"\nLooking for Python processes...")
    print(f"{'='*70}\n")

    # Find Python processes
    python_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                python_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if not python_procs:
        print("⚠️  No Python processes found!")
        return

    print(f"Found {len(python_procs)} Python process(es):")
    for proc in python_procs:
        try:
            print(f"  PID {proc.pid}: {' '.join(proc.cmdline()[:3])}")
        except:
            pass

    print(f"\n{'='*70}")
    print(f"MONITORING (Ctrl+C to stop)")
    print(f"{'='*70}\n")

    # Get CPU info
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()

    print(f"System: {cpu_count} CPUs @ {cpu_freq.current:.0f} MHz\n")

    # Column headers
    print(f"{'Time':<10} {'Total CPU':<12} {'Per-Core CPU':<50}")
    print(f"{'-'*70}")

    samples = []
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            # Get overall CPU
            total_cpu = psutil.cpu_percent(interval=interval)

            # Get per-core CPU
            per_core = psutil.cpu_percent(interval=0, percpu=True)

            # Get Python process CPU
            python_cpu_total = 0
            for proc in python_procs:
                try:
                    python_cpu_total += proc.cpu_percent(interval=0)
                except:
                    pass

            # Format per-core (show first 12 cores)
            core_str = ' '.join([f"{c:4.0f}" for c in per_core[:12]])

            # Print
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"{timestamp:<10} {total_cpu:5.1f}% ({python_cpu_total:5.1f}%)  {core_str}")

            # Store sample
            samples.append({
                'time': time.time() - start_time,
                'total_cpu': total_cpu,
                'python_cpu': python_cpu_total,
                'per_core': per_core
            })

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print("Monitoring stopped by user")

    # Analysis
    print(f"\n{'='*70}")
    print(f"ANALYSIS ({len(samples)} samples)")
    print(f"{'='*70}")

    if samples:
        avg_total = sum(s['total_cpu'] for s in samples) / len(samples)
        avg_python = sum(s['python_cpu'] for s in samples) / len(samples)
        max_total = max(s['total_cpu'] for s in samples)
        max_python = max(s['python_cpu'] for s in samples)

        print(f"\nOverall CPU Usage:")
        print(f"  Average: {avg_total:.1f}%")
        print(f"  Maximum: {max_total:.1f}%")

        print(f"\nPython Process CPU:")
        print(f"  Average: {avg_python:.1f}%")
        print(f"  Maximum: {max_python:.1f}%")

        # Per-core analysis
        avg_per_core = [sum(s['per_core'][i] for s in samples) / len(samples)
                        for i in range(min(cpu_count, 12))]

        print(f"\nPer-Core Average (first 12 cores):")
        for i, avg in enumerate(avg_per_core):
            bar = '█' * int(avg / 5)
            print(f"  Core {i:2d}: {avg:5.1f}% {bar}")

        # Diagnosis
        print(f"\n{'='*70}")
        print(f"DIAGNOSIS")
        print(f"{'='*70}")

        if avg_total < 30:
            print(f"⚠️  Very low CPU usage ({avg_total:.1f}%)")
            print(f"   Likely cause: Process is I/O bound or sleeping")
        elif avg_total < 60:
            print(f"⚠️  Low-moderate CPU usage ({avg_total:.1f}%)")
            print(f"   Expected for sequential quantum simulation")
            print(f"   PennyLane TorchLayer processes samples one-by-one")
        elif avg_total < 80:
            print(f"✓ Good CPU usage ({avg_total:.1f}%)")
            print(f"   System is working efficiently")
        else:
            print(f"✓ Excellent CPU usage ({avg_total:.1f}%)")
            print(f"   All cores being utilized")

        # Check core distribution
        core_variance = max(avg_per_core) - min(avg_per_core)
        if core_variance > 40:
            print(f"\n⚠️  Uneven core distribution (variance: {core_variance:.1f}%)")
            print(f"   Some cores very busy, others idle")
            print(f"   This is EXPECTED for quantum simulation!")
        else:
            print(f"\n✓ Even core distribution (variance: {core_variance:.1f}%)")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Default: monitor for 60 seconds
    duration = 60
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])

    print(f"\n⚠️  START YOUR TRAINING NOW!")
    print(f"This script will monitor CPU usage for {duration} seconds\n")
    print(f"Press Ctrl+C to stop early\n")

    time.sleep(2)  # Give user time to start training

    monitor_cpu_live(duration=duration, interval=0.5)
