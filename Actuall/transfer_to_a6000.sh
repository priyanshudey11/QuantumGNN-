#!/bin/bash
# Transfer script for A6000 server deployment

# ============================================================================
# CONFIGURATION - UPDATE THESE!
# ============================================================================

SERVER_USER="your_username"          # Change this!
SERVER_HOST="a6000-server.com"       # Change this!
SERVER_PATH="/home/user/quantum_gnn" # Change this!

# ============================================================================
# Files to transfer
# ============================================================================

echo "=================================="
echo "A6000 Server Transfer Script"
echo "=================================="
echo ""
echo "⚠️  BEFORE RUNNING:"
echo "    1. Update SERVER_USER, SERVER_HOST, SERVER_PATH in this script"
echo "    2. Ensure SSH key authentication is set up"
echo "    3. Make sure the data directory exists on the server"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

echo ""
echo "Creating remote directory..."
ssh ${SERVER_USER}@${SERVER_HOST} "mkdir -p ${SERVER_PATH}"

echo ""
echo "Transferring files..."

# Transfer the main training script
echo "  → train_a6000.py"
scp train_a6000.py ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/

# Transfer the notebook (if you prefer)
echo "  → compare_quantum_vs_classical_A6000.ipynb"
scp compare_quantum_vs_classical_A6000.ipynb ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/

# Transfer the model code
echo "  → drug_patient_qgnn/"
scp -r drug_patient_qgnn/ ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/

# Transfer monitoring script
echo "  → early_stopping_monitor.py"
scp early_stopping_monitor.py ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/

# Transfer setup guide
echo "  → A6000_SETUP_GUIDE.md"
scp A6000_SETUP_GUIDE.md ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/

echo ""
echo "✓ Transfer complete!"
echo ""
echo "=================================="
echo "Next steps:"
echo "=================================="
echo ""
echo "1. SSH into server:"
echo "   ssh ${SERVER_USER}@${SERVER_HOST}"
echo ""
echo "2. Navigate to project:"
echo "   cd ${SERVER_PATH}"
echo ""
echo "3. Update DATA_DIR in train_a6000.py:"
echo "   nano train_a6000.py  # Edit line 22"
echo ""
echo "4. Start training in tmux:"
echo "   tmux new -s quantum"
echo "   conda activate your_env"
echo "   python train_a6000.py"
echo "   # Detach: Ctrl+B then D"
echo ""
echo "5. Monitor progress (in another terminal):"
echo "   tail -f ${SERVER_PATH}/a6000_training.log"
echo ""
echo "6. Transfer results back when complete:"
echo "   scp -r ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/a6000_results/ ./"
echo ""
