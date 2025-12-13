.PHONY: help install data train evaluate clean

help:
	@echo "Quantum Drug-Protein Interaction Pipeline"
	@echo ""
	@echo "Available commands:"
	@echo "  make install           - Install dependencies"
	@echo "  make data              - Generate sample data"
	@echo "  make train             - Train full quantum model"
	@echo "  make train-classical   - Train classical baseline"
	@echo "  make evaluate          - Evaluate trained model"
	@echo "  make clean             - Clean outputs"
	@echo "  make help              - Show this help"

install:
	pip install -r requirements.txt

data:
	python scripts/generate_sample_data.py

train:
	python run_pipeline.py --config configs/default.yaml --model-type full

train-classical:
	python run_pipeline.py --config configs/default.yaml --model-type classical

evaluate:
	@echo "Run specific evaluation with:"
	@echo "  python -m src.evaluate --ckpt outputs/model_full.pt --config configs/default.yaml"

clean:
	rm -rf outputs/* figures/* reports/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

.DEFAULT_GOAL := help
