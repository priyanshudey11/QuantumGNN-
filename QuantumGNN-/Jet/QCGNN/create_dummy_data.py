#!/usr/bin/env python3

import os
import h5py
import numpy as np

# Create dataset directories
os.makedirs('dataset/jetnet', exist_ok=True)
os.makedirs('dataset/top', exist_ok=True)

# Create JetNet dummy data
print("Creating JetNet dummy data...")
channels = ['q', 'g', 't', 'w', 'z']
num_jets = 5000  # Create more data to account for filtering
num_particles = 30
num_features = 4  # (eta_rel, phi_rel, pt_rel, mask)

for channel in channels:
    filepath = f'dataset/jetnet/{channel}.hdf5'
    with h5py.File(filepath, 'w') as f:
        # particle_features: (N, 30, 4)
        particle_features = np.random.randn(num_jets, num_particles, num_features)
        particle_features[..., 2] = np.abs(particle_features[..., 2]) * 0.1 + 0.05  # pt_rel: 0.05-0.15 (above threshold)
        particle_features[..., 3] = (np.random.rand(num_jets, num_particles) > 0.5).astype(float)  # mask (50% particles)

        # jet_features: (N, 4) -> (pt, eta, mass, num_particles)
        jet_features = np.random.randn(num_jets, 4)
        jet_features[:, 0] = np.random.uniform(800, 1200, size=num_jets)  # pt: 800-1200 (matches config)
        jet_features[:, 2] = np.abs(jet_features[:, 2]) * 50  # mass positive
        jet_features[:, 3] = np.random.randint(4, 17, size=num_jets)  # num particles: 4-16 (matches config)

        f.create_dataset('particle_features', data=particle_features)
        f.create_dataset('jet_features', data=jet_features)

    print(f"  Created {filepath}")

