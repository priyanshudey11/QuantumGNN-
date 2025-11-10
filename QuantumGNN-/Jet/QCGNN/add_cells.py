import json

with open('quick_train_all.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Paper comparison cells
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Figure 3: Comparison with Paper's Published Results\n\nCompare quick_train results (250 samples/class, 1 epoch) with the paper's full results (25,000 samples/class, ~100 epochs)."
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "# Paper's published results for JetNet dataset (from arXiv 2403.04990)\n# These are from full training: 25,000 samples/class, ~100 epochs\npaper_results = {\n    'QCGNN-6': {'auc': 0.822, 'accuracy': 0.543, 'auc_std': 0.003, 'acc_std': 0.006},\n    'QCGNN-3': {'auc': 0.796, 'accuracy': 0.505, 'auc_std': 0.009, 'acc_std': 0.014},\n    'MPGNN': {'auc': 0.903, 'accuracy': 0.683, 'auc_std': 0.002, 'acc_std': 0.007},\n    'PFN': {'auc': 0.900, 'accuracy': 0.675, 'auc_std': 0.003, 'acc_std': 0.005},\n    'ParT': {'auc': 0.889, 'accuracy': 0.656, 'auc_std': 0.002, 'acc_std': 0.006},\n    'PNet': {'auc': 0.896, 'accuracy': 0.669, 'auc_std': 0.003, 'acc_std': 0.004},\n}\n\nprint(\"Paper's Published Results (JetNet, 25K samples/class, ~100 epochs):\")\nprint(\"=\"*70)\nfor model, metrics in paper_results.items():\n    print(f\"{model:12s}: AUC = {metrics['auc']:.3f} ± {metrics['auc_std']:.3f}, \"\n          f\"Acc = {metrics['accuracy']:.3f} ± {metrics['acc_std']:.3f}\")\nprint(\"=\"*70)"
    }
]

nb['cells'].extend(new_cells)

with open('quick_train_all.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Added {len(new_cells)} cells. Total: {len(nb['cells'])}")
