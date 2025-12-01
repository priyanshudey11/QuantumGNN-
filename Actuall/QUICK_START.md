# Quick Start Guide - Choose Your Path

## 🚀 Fastest: A6000 Server (2-3 hours)

```bash
# 1. Edit transfer script
nano transfer_to_a6000.sh
# Update: SERVER_USER, SERVER_HOST, SERVER_PATH

# 2. Transfer files
./transfer_to_a6000.sh

# 3. SSH to server
ssh user@a6000-server

# 4. Update data path
cd /path/to/quantum_gnn
nano train_a6000.py  # Edit line 22: DATA_DIR = "/your/path/to/data"

# 5. Run training in background
tmux new -s quantum
conda activate your_env
python train_a6000.py
# Detach: Ctrl+B then D

# 6. Monitor from local machine
./monitor_a6000_remote.sh  # Edit config first!
# Or manually: ssh user@server "tail -f /path/to/a6000_training.log"

# 7. Get results when done
scp -r user@server:/path/to/quantum_gnn/a6000_results/ ./
```

**Time:** 2-3 hours | **Setup:** 15 min

---

## 💻 Local Optimized (6-8 hours)

```bash
cd /home/priyanshu/QuantumGNN-/Actuall
jupyter lab
# Open: compare_quantum_vs_classical_OPTIMIZED.ipynb
# Run all cells
```

**Time:** 6-8 hours | **Setup:** 0 min

---

## 🔄 Resume DEBUG (20+ hours)

```bash
cd /home/priyanshu/QuantumGNN-/Actuall
jupyter lab
# Open: RESUME_training.ipynb
# Run all cells
```

**Time:** 20+ hours | **Setup:** 0 min

---

## 📊 Files Overview

**Training Scripts:**
- `train_a6000.py` - A6000 server script (recommended)
- `compare_quantum_vs_classical_A6000.ipynb` - A6000 notebook
- `compare_quantum_vs_classical_OPTIMIZED.ipynb` - Local optimized
- `RESUME_training.ipynb` - Continue from checkpoint

**Setup & Monitoring:**
- `transfer_to_a6000.sh` - Transfer files to server
- `monitor_a6000_remote.sh` - Monitor remote training
- `early_stopping_monitor.py` - Watch training progress
- `A6000_SETUP_GUIDE.md` - Detailed server setup

**Documentation:**
- `TRAINING_OPTIONS.md` - Compare all 3 options
- `quantum_advantage_math.md` - Mathematical proof
- `QUICK_START.md` - This file

---

## ⚡ Recommended: A6000 Server

**Why A6000?**
- 10× faster (2-3 hours vs 20+ hours)
- Full dataset (2000 drugs)
- Full quantum power (3 layers)
- Multi-GPU for classical model
- Best demonstration of quantum advantage

**Prerequisites:**
- SSH access to A6000 server
- Data directory accessible on server
- Conda environment with: torch, pennylane, rdkit, scikit-learn

---

## 📈 What to Expect

**Quantum Model:**
- ~15-20 min/epoch (CPU quantum simulation)
- Expected best AUC: 0.75-0.80
- Will likely converge in 30-50 epochs

**Classical Model:**
- ~30 sec/epoch (multi-GPU accelerated)
- Expected best AUC: 0.72-0.77
- Will likely converge in 40-60 epochs

**Results will show:**
- Quantum vs Classical AUC comparison
- Training curves (loss, accuracy, F1)
- Statistical significance of differences

---

## 🆘 Troubleshooting

**"Module not found" on server:**
```bash
conda activate your_env
pip install torch pennylane rdkit scikit-learn matplotlib pandas
```

**"CUDA out of memory" on A6000:**
Edit `train_a6000.py`, change `BATCH_SIZE = 2048` to `BATCH_SIZE = 1024`

**Training seems stuck:**
Check if quantum circuit is running: `top` should show high CPU usage (100% on 1 core)

**Can't connect to server:**
Verify SSH access: `ssh user@server "echo Connected!"`

---

## 📞 Quick Commands

**Check GPU:**
```bash
nvidia-smi
```

**Check training progress:**
```bash
tail -f a6000_training.log
# Or: python early_stopping_monitor.py
```

**Check if training is running:**
```bash
ps aux | grep python
# Or: tmux ls && tmux attach -t quantum
```

**Kill training if needed:**
```bash
tmux attach -t quantum
# Then: Ctrl+C
```

---

## ✅ Next Steps

1. Choose your option (A6000 recommended)
2. Follow the commands above
3. Monitor training progress
4. Analyze results when complete
5. Compare quantum vs classical performance

Good luck! 🚀
