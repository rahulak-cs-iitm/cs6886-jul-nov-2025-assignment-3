# cs6886-jul-nov-2025-assignment-3
#===================================================================
# Training and Compressing MobileNetV2 for CIFAR-10 Classification
#-------------------------------------------------------------------

This GitHub Repository was created as part of Assignment-3 for CS6886 Systems Engineering for Deep Learning Course at IIT Madras. Training was done on an A5000 GPU with 24GB Graphics Memory.

The repo contains the following action files and helper files:

**Environment & Dependency:** `requirements.txt`

**Seed Setting:**
A `set_seed` function has been added in `utils.py`. A fixed seed value of `47` is used for reproducibility.

### Action Files
---

**1. Training Baseline for MobileNetV2 on CIFAR-10**
   - **run:**
     ```
     make train
     ```
   - **outputs:**
     ```
     checkpoints/mobilenetv2_cifar10.pth (Saved Weights)
     ```
   - **logs:**
     ```
     logs/mobilenetv2_training_log.txt
     ```
   - **plots:**
     ```
     plots/training_loss_plot.png
     plots/accuracy_plot.png
     ```

**2. Evaluate the model** (takes weights from `mobilenetv2_cifar10.pth`)
   - **run:**
     ```
     make eval
     ```
   - **logs:**
     ```
     logs/log_eval.txt
     ```

**3. Magnitude Pruning with Fine Tuning** (Global, Unstructured)
   - **run:**
     ```
     make prune
     ```
   - **outputs:**
     ```
     checkpoints/mobilenetv2_cifar10_pruned.pth
     ```
   - **logs:**
     ```
     logs/log_iterative_pruning.txt
     ```

**4. Evaluate Sparsity and accuracy of pruned model**
   - **run:**
     ```
     make eval_sparsity
     ```
   - **logs:**
     ```
     logs/log_eval_sparsity.txt
     ```

**5. Symmetric Quantization for Weights & Activations**
   - **run:**
     ```
     make quant
     ```
   - **outputs:**
     ```
     checkpoints/mobilenetv2_pruned_symmetric_quantized_16bit.pth
     checkpoints/mobilenetv2_pruned_symmetric_quantized_8bit.pth
     checkpoints/mobilenetv2_pruned_symmetric_quantized_4bit.pth
     ```
   - **logs:**
     ```
     logs/log_symmetric_quantization.txt
     ```
   - **plots:**
      ```
      plots/symmetric_quantization_results.png
      ```

**6. Generate and evaluate int8 model**
   - **run:**
     ```
     make int8_model
     ```
   - **outputs:**
     ```
     checkpoints/int8_mobilenetv2_pruned_symmetric_quantized_8bit.pth
     ```
   - **logs:**
     ```
     logs/log_gen_int8_model.txt
     ```

**7. Sparse Compression** (stores non-zero weight values and indices)
   - **run:**
     ```
     make compress
     ```
   - **outputs:**
     ```
     ./checkpoints/mobilenetv2_sparse_compression.pth
     ```
   - **logs:**
     ```
     ./logs/log_sparse_compression.txt
     ```

**8. Sparse Model Analysis** (profiles storage use in sparse model)
   - **run:**
     ```
     make analyze_sparse_model
     ```
   - **logs:**
     ```
     ./logs/log_analyze_sparse_model.txt
     ```

**9. Compression Statistics** (final compression/accuracy stats)
   - **run:**
     ```
     make compression_stats
     ```
   - **logs:**
     ```
     ./logs/log_compression_stats.txt
     ```

**10. Configurable Compression Pipeline** (configurable bits)
   - **run:**
     ```
     make configurable_compression WEIGHT_BITS=<bits> ACTIVATION_BITS=<bits>
     ```
   - **outputs:**
     ```
     checkpoints/configurable_compression/mobilenetv2_quantized_w<bits>bit_a<bits>bit.pth
     ```
   - **logs:**
     ```
     logs/configurable_compression/log_configurable_compression_w<bits>bits_a<bits>bits.txt
     ```

**11. Generate WandB Parallel Coordinates Chart**
   - **run:**
     ```
     make wandb_gen
     ```
   - **logs:**
     ```
     logs/wandb_stats/wandb_compression_results_<DATE>_<TIME>.csv
     ```

### Helper Files
---
**1. `dataloader.py`**
   - Loads CIFAR-10 data with transforms and normalization.
   - **outputs:** `data/cifar-10-batches-py`

**2. `utils.py`**
   - Contains helper functions for setting seed, evaluation, etc.

**3. `Makefile`**
   - Contains commands for `train`, `evaluate`, `prune`, `quantize`, `clean`, etc.
