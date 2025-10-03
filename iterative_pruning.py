# iterative_pruning.py
# Prunes and fine tunes the model, pruning X% of weights everytime
# The moment accuracy drops below a threshold, pruning stops

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import os
import copy
from utils import set_seed, get_model_architecture, evaluate, calculate_sparsity_weights, calculate_theoretical_compression, get_all_weights
from dataloader import get_cifar10
import time


# Iterative Magnitude pruning (Global, unstructured)
def prune_model_by_magnitude(model, prune_ratio):
    """
    Prunes the model in-place by removing a prune ration percentage of the remaining non-zero weights.
    """
    all_weights = get_all_weights(model)

    # 1. Collect all non-zero weight magnitudes
    nonzero_weights = []
    for weight in all_weights:
        nonzero_weights.append(weight.data[weight.data.nonzero(as_tuple=True)].abs())

    if not nonzero_weights:
        print("No non-zero weights left to prune.")
        return

    flat_nonzero_weights = torch.cat(nonzero_weights)
    
    # 2. Determine no of weights to be pruned based only on the non-zero weights
    k = int(prune_ratio * flat_nonzero_weights.numel())

    if k == 0:
        print(f"Pruning ratio ({prune_ratio}) is too small for the number of remaining weights. No new weights pruned.")
        return

    # Set threshold value
    threshold = torch.kthvalue(flat_nonzero_weights, k).values

    # 3. Apply the pruning mask
    with torch.no_grad():
        for weight in all_weights:
            mask = weight.abs() > threshold
            weight.mul_(mask) # in-place multiplication to set weights with mask = 0 to 0

    print(f"Pruned {prune_ratio*100:.2f}% of remaining weights. New threshold: {threshold.item()}")

def finetune_model(model, train_loader, criterion, optimizer, device, epochs):
    """
    Finetunes the model keeping pruned weights zero (using same mask after fine-tune)
    """
    # Create a mask from the current model's sparsity
    masks = []
    with torch.no_grad():
        for param in get_all_weights(model):
            masks.append(param.data != 0)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Apply prev calculated mask to retain sparsity 
            with torch.no_grad():
                for param, mask in zip(get_all_weights(model), masks):
                    param.data.mul_(mask)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"  Finetune Epoch {epoch+1}/{epochs} -> Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

if __name__ == "__main__":
    set_seed(47)
    # --- Configuration ---
    ACCURACY_THRESHOLD = 94.0 # If accuracy goes below this, stop
    PRUNE_ITERATIONS = 80
    PRUNE_RATIO_PER_ITERATION = 0.2 # Prune 20% of remaining weights each time
    FINETUNE_EPOCHS = 10
    LEARNING_RATE = 0.001 # Using a smaller learning rate for finetuning (0.07 was used for training)
    BATCH_SIZE = 128
    MODEL_CHECKPOINT = "./checkpoints/mobilenetv2_cifar10.pth"
    PRUNED_MODEL_SAVE_PATH = "./checkpoints/mobilenetv2_cifar10_pruned.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_cifar10(batchsize=BATCH_SIZE)

    # --- Load Pre-trained Model ---
    model = get_model_architecture()
    
    if not os.path.exists(MODEL_CHECKPOINT):
        print(f"Error: Model checkpoint not found at {MODEL_CHECKPOINT}")
        print("Please ensure you have trained the model using main.py first.")
    else:
        print(f"Loading pre-trained model from {MODEL_CHECKPOINT}")
        model.load_state_dict(torch.load(MODEL_CHECKPOINT,weights_only=True))
        model.to(device)

        # --- Evaluate Initial Model ---
        initial_accuracy = evaluate(model, test_loader, device)
        print(f"\nInitial Test Accuracy: {initial_accuracy:.2f}%")
        print(f"Initial Sparsity: {calculate_sparsity_weights(model):.2f}%\n")

        # --- Iterative Pruning and Finetuning ---
        pruned_model = copy.deepcopy(model)
        optimizer = torch.optim.SGD(pruned_model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        # Saving the last good model
        last_good_model_state = None
        
        for i in range(PRUNE_ITERATIONS):
            print(f"--- Iteration {i+1}/{PRUNE_ITERATIONS} ---")
            start_time = time.time()
            
            # Prune the model
            prune_model_by_magnitude(pruned_model, PRUNE_RATIO_PER_ITERATION)
            
            # Finetune the model
            print(f"Finetuning for {FINETUNE_EPOCHS} epochs with LR={LEARNING_RATE}...")
            finetune_model(pruned_model, train_loader, criterion, optimizer, device, epochs=FINETUNE_EPOCHS)
            
            # Evaluate the pruned model
            accuracy = evaluate(pruned_model, test_loader, device)
            sparsity = calculate_sparsity_weights(pruned_model)
            end_time = time.time()
            duration = end_time - start_time
            print(f"  Iteration {i+1} -> Duration: {duration:.2f}s | Test Accuracy: {accuracy:.2f}%, Global Sparsity: {sparsity:.2f}%\n")
            # Stopping if accuracy goes below threshold
            if accuracy < ACCURACY_THRESHOLD:
                print(f"\nAccuracy {accuracy:.2f}% fell below the threshold of {ACCURACY_THRESHOLD}%.")
                print("Stopping the pruning process.")
                # Restore the last good model to pruned_model, else it will save the one which went below threshold
                if last_good_model_state is not None:
                  pruned_model.load_state_dict(last_good_model_state)
                  print("Restored model to last good state.")
                break  # Exit the loop
            else:
                # Accuracy above threshold, so we save model state
                last_good_model_state = copy.deepcopy(pruned_model.state_dict())
                print(f"  Accuracy is above threshold. Saving model state and continuing.")

        # --- Final Evaluation and Save ---
        print("--- Completed Iterative Pruning ---")
        final_accuracy = evaluate(pruned_model, test_loader, device)
        final_sparsity = calculate_sparsity_weights(pruned_model)
        
        print(f"Initial Accuracy: {initial_accuracy:.2f}%")
        print(f"Final Accuracy:   {final_accuracy:.2f}%")
        print(f"Final Sparsity:   {final_sparsity:.2f}%")

        original_size, compressed_size, ratio = calculate_theoretical_compression(pruned_model)
        
        print("\n--- Theoretically Possible Compression ---")
        print(f"Original Model Size : {original_size / 1e6:.2f} MB")
        print(f"Compressed Model Size (if zeros removed): {compressed_size / 1e6:.2f} MB")
        print(f"Theoretically Possible Compression Ratio: {ratio:.2f}x")
        
        # Save the final pruned model
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save(pruned_model.state_dict(), PRUNED_MODEL_SAVE_PATH)
        print(f"\nFinal pruned model saved to {PRUNED_MODEL_SAVE_PATH}")
