#===================================================================
# Training and Compressing MobileNetV2 for CIFAR-10 Classification
#-------------------------------------------------------------------

This GitHub Repository was created as part of Assignment-3 for CS6886 Systems Engineering for Deep Learning Course at IIT Madras. Training was done on an A5000 GPU with 24GB Graphics Memory.

The repo contains following action files and helper files:

Environment & Dependency:

Seed Setting:
  set_seed function added in utils. Fixed seed value 47 used for reproducibility.

Action Files
------------
  1. Training Baseline for MobileNetV2 on CIFAR-10 (train_mobilenetv2_cifar10)
      run     -> make train
      outputs -> checkpoints/mobilenetv2_cifar10.pth (Saved Weights)
      logs    -> logs/mobilenetv2_training_log.txt
      plots   -> plots/training_loss_plot.png
                 plots/accuracy_plot.png
  
  2. Evaluate the model (takes weights from mobilenetv2_cifar10.pth)
      run     -> make eval
      logs    -> logs/log_eval.txt
  
  3. Magnitude Pruning with Fine Tuning (Global, Unstructured Pruning)
      run     -> make prune
      outputs -> checkpoints/mobilenetv2_cifar10_pruned.pth
      logs    -> logs/log_iterative_pruning.txt
  
  4. Evaluate Sparsity and accuracy of pruned model
      run     -> make eval_sparsity
      logs    -> logs/log_eval_sparsity.txt
  
  5. Symmetric Quantization for both Weights & Activations
      run     -> make quant
      outputs -> checkpoints/mobilenetv2_pruned_symmetric_quantized_16bit.pth
                 checkpoints/mobilenetv2_pruned_symmetric_quantized_8bit.pth
                 checkpoints/mobilenetv2_pruned_symmetric_quantized_4bit.pth
      logs    -> logs/log_symmetric_quantization.txt
                 plots/symmetric_quantization_results.png

  6. Generate and evaluate int8 model
      run     -> make int8_model
      outputs -> checkpoints/int8_mobilenetv2_pruned_symmetric_quantized_8bit.pth
      logs    -> logs/log_gen_int8_model.txt

  7. Sparse Compression (simple compression storing values and indices of non-zero weights)
      run     -> make compress
      outputs -> ./checkpoints/mobilenetv2_sparse_compression.pth
      logs    -> ./logs/log_sparse_compression.txt

  8. Sparse Model Analysis (to profile storage use in sparse model)
      run     -> make analyze_sparse_model
      logs    -> ./logs/log_analyze_sparse_model.txt

  9. Compression Statistics (final compression/accuracy stats)
      run     -> make compression_stats
      logs    -> ./logs/log_compression_stats.txt

  10. Configurable Compression Pipeline (configurable bits for weights/activations)
      run     ->  make configurable_compression WEIGHT_BITS=<weight_bits> ACTIVATION_BITS=<activation_bits>
      outputs -> checkpoints/configurable_compression/mobilenetv2_quantized_w<weight_bits>bit_a<activation_bits>bit.pth
      logs    -> logs/configurable_compression/log_configurable_compression_w<weight_bits>bits_a<activation_bits>bits.txt

  11. Generate WandB Parallel Coordinates Chart (viewable online at wandb.com)
      run     ->  make wandb_gen
      logs    -> logs/wandb_stats/wandb_compression_results_<DATE>_<TIME>.csv

Helper Files
  1. dataloader.py : To load CIFAR-10 data with transforms and normalization for train and test
    outputs -> data/cifar-10-batches-py (CIFAR-10 database)
  2. utils.py : Helper functions for setting seed, evaluate, get_model_architecture
  3. Makefile for train, evaluate, prune, quantize, huffman-encode, evaluate-encoding, clean, clean_all

  
