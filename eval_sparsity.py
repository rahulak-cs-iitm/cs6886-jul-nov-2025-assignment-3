import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import os
import time
from utils import set_seed, save_weights_to_file, get_model_architecture, evaluate, calculate_sparsity
from dataloader import get_cifar10

if __name__ == "__main__":
    set_seed(47)
    # set_seed(0)
    BATCH_SIZE = 128
    # BATCH_SIZE = 256
    MODEL_CHECKPOINT = "./checkpoints/mobilenetv2_cifar10.pth"
    PRUNED_MODEL_SAVE_PATH = "./checkpoints/mobilenetv2_cifar10_pruned.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_cifar10(batchsize=BATCH_SIZE)
    
    # Original Model
    print("\n-------------- Evaluating Original Model -------------")
    model_original = get_model_architecture()
    model_original.load_state_dict(torch.load(MODEL_CHECKPOINT,weights_only=True))
    # save_weights_to_file(model_original, "model_original_weights.txt")
    model_original.to(device)
    original_accuracy = evaluate(model_original, test_loader, device)
    print(f"Original accuracy: {original_accuracy:.2f}%")

    # Pruned Model
    print("\n-------------- Evaluating Pruned Model -------------")
    model_pruned = get_model_architecture()
    model_pruned.load_state_dict(torch.load(PRUNED_MODEL_SAVE_PATH,weights_only=True))
    # save_weights_to_file(model_pruned, "model_pruned_weights.txt")
    model_pruned.to(device)
    pruned_accuracy = evaluate(model_pruned, test_loader, device)
    print(f"Pruned accuracy: {pruned_accuracy:.2f}%")

    accuracy_difference = original_accuracy - pruned_accuracy
    print(f"\nAccuracy Difference (Original - Pruned): {accuracy_difference:.2f}%\n")

    # Calculating Sparsity
    print("\n-------------- Calculating Sparsity -------------\n")
    state_dict_original = torch.load(MODEL_CHECKPOINT,weights_only=True)
    sparsity_original, total_original, zeros_original = calculate_sparsity(state_dict_original)
    state_dict_pruned = torch.load(PRUNED_MODEL_SAVE_PATH,weights_only=True)
    sparsity_pruned, total_pruned, zeros_pruned = calculate_sparsity(state_dict_pruned)

    # --- Print Results ---
    print(f"--- Sparsity Comparison ---")
    print(f"Model 1 ({MODEL_CHECKPOINT}):")
    print(f"  Total Weight Elements: {total_original:,}")
    print(f"  Zero Weight Elements:  {zeros_original:,}")
    print(f"  Overall Sparsity:      {sparsity_original:.2f}%")
    
    print(f"\nModel 2 ({PRUNED_MODEL_SAVE_PATH}):")
    print(f"  Total Weight Elements: {total_pruned:,}")
    print(f"  Zero Weight Elements:  {zeros_pruned:,}")
    print(f"  Overall Sparsity:      {sparsity_pruned:.2f}%")
    
    # --- Final Comparison ---
    sparsity_diff = sparsity_pruned - sparsity_original
    print(f"Sparsity Difference (Pruned - Original): {sparsity_diff:.2f} percentage points")
