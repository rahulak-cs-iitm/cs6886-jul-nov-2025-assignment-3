# ===============================================
# Makefile for ML Model Workflow
# Targets: train, evaluate, prune, all
# ===============================================

# Variables
PYTHON := python3

# --- PHONY Targets ---
.PHONY: all train eval prune eval_sparsity quant int8_model compress analyze_compression

# --- Main Targets ---

all: train evaluate # Default target: runs training and evaluation

## Runs the main training script. No arguments needed.
train:
	@echo "--- Starting Model Training (main.py) ---"
	$(PYTHON) train_mobilenetv2_cifar10.py

## Runs the evaluation script on the trained model.
eval:
	@echo "--- Starting Model Evaluation (eval.py) ---"
	$(PYTHON) eval.py | tee ./logs/log_eval.txt

## Runs the pruning script.
prune:
	@echo "--- Starting iterative pruning ---"
	$(PYTHON) iterative_pruning.py | tee ./logs/log_iterative_pruning.txt

## Runs the evaluation script to evaluate sparsity and accuracy of pruned model.
eval_sparsity:
	@echo "--- Starting Sparsity Evaluation (eval_sparsity.py) ---"
	$(PYTHON) eval_sparsity.py | tee ./logs/log_eval_sparsity.txt

## Runs symmetric quantization on the pruned model for 16/8/4 bits
quant:
	@echo "--- Starting Symmetric Quantization (symmetric_quantization.py) ---"
	$(PYTHON) symmetric_quantization.py | tee ./logs/log_symmetric_quantization.txt

## Runs symmetric quantization on the pruned model for 16/8/4 bits
int8_model:
	@echo "--- Generating int8 model (symmetric_quantization.py) ---"
	$(PYTHON) gen_int8_model.py | tee ./logs/log_gen_int8_model.txt

## Runs Sparse compression on the int8 model
compress:
	@echo "--- Starting Sparse Compression on int8 model (sparse_compression.py) ---"
	$(PYTHON) sparse_compression.py | tee ./logs/log_sparse_compression.txt

## Runs Sparse compression on the int8 model
analyze_sparse_model:
	@echo "--- Starting Analysis of Compressed Sparse Model (analyze_sparse_model.py) ---"
	$(PYTHON) analyze_sparse_model.py | tee ./logs/log_analyze_sparse_model.txt

## Runs Sparse compression on the int8 model
compression_stats:
	@echo "--- Starting Compression Statistics (compression_stats.py) ---"
	$(PYTHON) compression_stats.py | tee ./logs/log_compression_stats.txt

## Configurable Compression for different weight/activation bit widths 
# Usage: make configurable_compression WEIGHT_BITS=8 ACTIVATION_BITS=8
configurable_compression:
	@echo "--- Starting Configurable Compression (configurable_compression.py) ---"
	($(PYTHON) configurable_compression.py \
		--weight_bits $(WEIGHT_BITS) \
		--activation_bits $(ACTIVATION_BITS)) \
		| tee ./logs/configurable_compression/log_configurable_compression_w$(WEIGHT_BITS)bits_a$(ACTIVATION_BITS)bits.txt

## Runs WandB Parallel Coordinates Generation script 
wandb_gen:
	@echo "--- Starting WandB Parallel Coordinates Generation (gen_wandb_parallel_coords_chart.py) ---"
	$(PYTHON) gen_wandb_parallel_coords_chart.py

# --- Utility Targets ---

## Cleans up common compiled files and model checkpoints (customize as needed)
clean:
	@echo "--- Cleaning up project files ---"
	rm -f *.pyc
	rm -rf __pycache__
	# Example: remove saved model/checkpoint files
	# rm -f checkpoint/*.pth model_output.h5

clean_all:
	@echo "--- Cleaning up EVERYTHING !! ---"
	rm -f *.pyc
	rm -rf __pycache__
	rm -rf checkpoints
	rm -rf data
	rm -rf logs
