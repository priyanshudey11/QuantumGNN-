#!/usr/bin/env python3
"""
Test the hardware detection and optimization system.
Run this to see what hardware is detected and what configuration is recommended.
"""

from hardware_optimizer import setup_environment

if __name__ == '__main__':
    print("Testing automatic hardware detection...\n")
    hw_info, config = setup_environment()

    print("\n" + "="*70)
    print("CONFIGURATION DETAILS")
    print("="*70)
    print(f"\nThis configuration is optimized for:")
    print(f"  - {hw_info['cpu_brand']}")
    print(f"  - {hw_info['device_name']}")
    print(f"\nRecommended settings:")
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    print("="*70)
