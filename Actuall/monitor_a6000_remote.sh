#!/bin/bash
# Remote monitoring script for A6000 training

# ============================================================================
# CONFIGURATION - UPDATE THESE!
# ============================================================================

SERVER_USER="your_username"
SERVER_HOST="a6000-server.com"
SERVER_PATH="/home/user/quantum_gnn"

# ============================================================================
# Monitoring Functions
# ============================================================================

show_menu() {
    echo ""
    echo "=================================="
    echo "A6000 Training Monitor"
    echo "=================================="
    echo "1. View live log (tail -f)"
    echo "2. Check latest status"
    echo "3. View GPU usage"
    echo "4. Check training progress (JSON)"
    echo "5. List result files"
    echo "6. Download results"
    echo "7. SSH to server"
    echo "8. Exit"
    echo "=================================="
    echo -n "Choose option: "
}

view_live_log() {
    echo "Viewing live log (Ctrl+C to stop)..."
    ssh ${SERVER_USER}@${SERVER_HOST} "tail -f ${SERVER_PATH}/a6000_training.log"
}

check_status() {
    echo "Latest training status:"
    echo ""
    ssh ${SERVER_USER}@${SERVER_HOST} "tail -20 ${SERVER_PATH}/a6000_training.log"
}

check_gpu() {
    echo "GPU Usage on A6000 server:"
    echo ""
    ssh ${SERVER_USER}@${SERVER_HOST} "nvidia-smi"
}

check_progress() {
    echo "Training Progress:"
    echo ""
    ssh ${SERVER_USER}@${SERVER_HOST} "cd ${SERVER_PATH} && python3 << 'EOPY'
import json
import os

results_dir = './a6000_results'

# Check quantum progress
quantum_file = os.path.join(results_dir, 'quantum_history.json')
if os.path.exists(quantum_file):
    with open(quantum_file, 'r') as f:
        h = json.load(f)
    epochs = len(h['val_auc'])
    best_auc = max(h['val_auc'])
    current_auc = h['val_auc'][-1]
    print(f'Quantum Model:')
    print(f'  Epochs completed: {epochs}/100')
    print(f'  Current AUC: {current_auc:.4f}')
    print(f'  Best AUC: {best_auc:.4f}')
else:
    print('Quantum: Not started yet')

print()

# Check classical progress
classical_file = os.path.join(results_dir, 'classical_history.json')
if os.path.exists(classical_file):
    with open(classical_file, 'r') as f:
        h = json.load(f)
    epochs = len(h['val_auc'])
    best_auc = max(h['val_auc'])
    current_auc = h['val_auc'][-1]
    print(f'Classical Model:')
    print(f'  Epochs completed: {epochs}/100')
    print(f'  Current AUC: {current_auc:.4f}')
    print(f'  Best AUC: {best_auc:.4f}')
else:
    print('Classical: Not started yet')
EOPY
"
}

list_results() {
    echo "Result files on server:"
    echo ""
    ssh ${SERVER_USER}@${SERVER_HOST} "ls -lh ${SERVER_PATH}/a6000_results/ 2>/dev/null || echo 'No results yet'"
}

download_results() {
    echo "Downloading results from server..."
    echo ""

    LOCAL_DIR="./a6000_results_$(date +%Y%m%d_%H%M%S)"
    mkdir -p ${LOCAL_DIR}

    scp -r ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/a6000_results/* ${LOCAL_DIR}/
    scp ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/a6000_training.log ${LOCAL_DIR}/

    echo ""
    echo "✓ Results downloaded to: ${LOCAL_DIR}"
    ls -lh ${LOCAL_DIR}
}

ssh_connect() {
    echo "Connecting to server..."
    ssh ${SERVER_USER}@${SERVER_HOST} "cd ${SERVER_PATH} && exec bash"
}

# ============================================================================
# Main Loop
# ============================================================================

if [ "$SERVER_USER" == "your_username" ] || [ "$SERVER_HOST" == "a6000-server.com" ]; then
    echo "⚠️  ERROR: Please update SERVER_USER, SERVER_HOST, and SERVER_PATH in this script!"
    exit 1
fi

while true; do
    show_menu
    read choice

    case $choice in
        1) view_live_log ;;
        2) check_status ;;
        3) check_gpu ;;
        4) check_progress ;;
        5) list_results ;;
        6) download_results ;;
        7) ssh_connect ;;
        8) echo "Goodbye!"; exit 0 ;;
        *) echo "Invalid option" ;;
    esac

    echo ""
    read -p "Press Enter to continue..."
done
