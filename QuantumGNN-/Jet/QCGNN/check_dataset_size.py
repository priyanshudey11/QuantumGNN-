"""Check how many events are available in each JetNet channel after preprocessing."""

import yaml
from source.data.opendata import JetNetEvents

# Load config
with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Dataset settings
dataset_config = {}
dataset_config.update(config['Data'])
dataset_config.update(config['JetNet'])

# Check each channel
channels = ['q', 'g', 't', 'w', 'z']
print("\nAvailable events per channel after preprocessing:")
print("-" * 60)

for channel in channels:
    events_obj = JetNetEvents(channel=channel, **dataset_config)
    num_available = len(events_obj.events['fatjet_pt'])
    print(f"Channel '{channel}': {num_available:,} events")

print("-" * 60)

# Calculate how many are needed
num_train = config['Data']['num_train']
num_valid = config['Data']['num_valid']
num_test = config['Data']['num_test']
num_needed = num_train + num_valid + num_test

print(f"\nRequested per channel: {num_needed:,} events")
print(f"  - Train: {num_train:,}")
print(f"  - Valid: {num_valid:,}")
print(f"  - Test:  {num_test:,}")
