#!/usr/bin/env python3
"""
Real-time early stopping monitor for overnight runs.
Shows when early stopping is likely to trigger.
"""

import json
import time
import os
from datetime import datetime

RESULTS_DIR = "./overnight_comparison_results"
PATIENCE = 15

def monitor():
    """Monitor training progress and predict early stopping."""

    print("🔍 Early Stopping Monitor")
    print("="*60)
    print(f"Watching: {RESULTS_DIR}")
    print(f"Patience: {PATIENCE} epochs")
    print(f"Press Ctrl+C to exit\n")

    last_quantum_epoch = 0
    last_classical_epoch = 0

    while True:
        try:
            # Check Quantum progress
            quantum_file = os.path.join(RESULTS_DIR, "quantum_history.json")
            if os.path.exists(quantum_file):
                with open(quantum_file, 'r') as f:
                    quantum_history = json.load(f)

                epochs = len(quantum_history['val_auc'])
                if epochs > last_quantum_epoch:
                    last_quantum_epoch = epochs

                    # Find best AUC and when
                    val_auc = quantum_history['val_auc']
                    best_auc = max(val_auc)
                    best_epoch = val_auc.index(best_auc) + 1
                    current_auc = val_auc[-1]

                    # Calculate patience counter
                    patience_counter = epochs - best_epoch

                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] QUANTUM Model")
                    print(f"  Epoch: {epochs}/100")
                    print(f"  Current AUC: {current_auc:.4f}")
                    print(f"  Best AUC: {best_auc:.4f} (epoch {best_epoch})")
                    print(f"  Patience: {patience_counter}/{PATIENCE}", end="")

                    if patience_counter > 0:
                        remaining = PATIENCE - patience_counter
                        if remaining <= 5:
                            print(f" ⚠️  {remaining} epochs until stop!")
                        else:
                            print(f" ({remaining} epochs remaining)")
                    else:
                        print(" ✓ Improving!")

                    # Trend
                    if epochs >= 5:
                        recent_trend = val_auc[-1] - val_auc[-5]
                        trend_icon = "📈" if recent_trend > 0 else "📉"
                        print(f"  5-epoch trend: {recent_trend:+.4f} {trend_icon}")

            # Check Classical progress
            classical_file = os.path.join(RESULTS_DIR, "classical_history.json")
            if os.path.exists(classical_file):
                with open(classical_file, 'r') as f:
                    classical_history = json.load(f)

                epochs = len(classical_history['val_auc'])
                if epochs > last_classical_epoch:
                    last_classical_epoch = epochs

                    val_auc = classical_history['val_auc']
                    best_auc = max(val_auc)
                    best_epoch = val_auc.index(best_auc) + 1
                    current_auc = val_auc[-1]
                    patience_counter = epochs - best_epoch

                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] CLASSICAL Model")
                    print(f"  Epoch: {epochs}/100")
                    print(f"  Current AUC: {current_auc:.4f}")
                    print(f"  Best AUC: {best_auc:.4f} (epoch {best_epoch})")
                    print(f"  Patience: {patience_counter}/{PATIENCE}", end="")

                    if patience_counter > 0:
                        remaining = PATIENCE - patience_counter
                        if remaining <= 5:
                            print(f" ⚠️  {remaining} epochs until stop!")
                        else:
                            print(f" ({remaining} epochs remaining)")
                    else:
                        print(" ✓ Improving!")

            time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    monitor()
