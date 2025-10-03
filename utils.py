import torch 
import torch.nn as nn
from torchvision.models import mobilenet_v2


def evaluate(model, testloader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate((testloader)):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
        final_acc = 100 * correct / total
    return final_acc

# For seed
import random
import numpy as np

# Function to set seed for reproducibility (default seed=47)
def set_seed(seed=47):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# For analysis of weight values
def save_weights_to_file(model, filename="weights.txt"):
    """
    Iterates through all weight parameters of a model and writes their
    names, shapes, and values to a text file.
    """
    print(f"Saving all model weights to {filename}...")
    with open(filename, 'w') as f:
        f.write("--- MODEL WEIGHTS ---\n\n")
        
        # Iterate through all named parameters
        for name, param in model.named_parameters():
            # Filter for weight tensors
            if 'weight' in name:
                f.write(f"--- Layer: {name} ---\n")
                f.write(f"Shape: {param.size()}\n\n")
                
                # Move tensor to CPU and convert to NumPy array for saving
                weight_data = param.data.cpu().numpy()
                
                # Use numpy.savetxt for a clean, readable format
                np.savetxt(f, weight_data.reshape(-1, weight_data.shape[-1]), fmt='%f')
                
                f.write("\n" + "="*60 + "\n\n")
                
    print(f"Successfully saved all weights to {filename}.")

def get_model_architecture():
    """Returns the base MobileNetV2 architecture for CIFAR-10."""
    model = mobilenet_v2(weights=None)
    model.features[0][0].stride = (1, 1)
    model.features[2].conv[1][0].stride = (1, 1)
    model.features[4].conv[1][0].stride = (1, 1)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 10)
    return model


# Used in iterative_pruning.py
def get_all_weights(model):
    """
    Gathers all weight parameters from the model's convolutional and linear layers.
    """
    weights = []
    for name, param in model.named_parameters():
        if "weight" in name and isinstance(model.get_submodule(name.rsplit('.', 1)[0]), (nn.Conv2d, nn.Linear)):
            weights.append(param)
    return weights

def calculate_sparsity_weights(model):
    """
    Calculates the global sparsity of the model.
    """
    all_weights = get_all_weights(model)
    total_weights = 0
    zero_weights = 0
    for weight_tensor in all_weights:
        total_weights += weight_tensor.nelement()
        zero_weights += torch.sum(weight_tensor == 0).item()
    
    sparsity = 100. * zero_weights / total_weights
    return sparsity

def calculate_theoretical_compression(model):
    """
    Calculates the theoretical size and compression ratio if zero-weights are not stored.
    This assumes parameters are stored as 32-bit floats (4 bytes).
    """
    total_params = 0
    nonzero_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            nonzero_params += torch.count_nonzero(param).item()
            
    # Calculate sizes in bytes
    original_size_bytes = total_params * 4
    compressed_size_bytes = nonzero_params * 4
    
    # Calculate the compression ratio
    compression_ratio = original_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else float('inf')
    
    return original_size_bytes, compressed_size_bytes, compression_ratio

# Used in models_check.py
def calculate_sparsity(model_state_dict):
    """
    Calculates the overall weight sparsity of a model's state dictionary
    """
    total_elements = 0
    zero_elements = 0

    for name, param in model_state_dict.items():
        # Check if the parameter name contains 'weight' and is not a mask (for pruned models)
        if 'weight' in name and 'mask' not in name:
            # Flatten the tensor to easily count zero elements
            tensor_flat = param.abs().flatten()
            
            # UNUSED # Count elements close to zero (1e-8 common threshold)
            # UNUSED # zeros = (tensor_flat < 1e-8).sum().item()
            # Using absolute zero weights
            zeros = (tensor_flat == 0).sum().item()
            
            total_elements += tensor_flat.numel()
            zero_elements += zeros

    if total_elements == 0:
        return 0.0

    sparsity = (zero_elements / total_elements) * 100
    return sparsity, total_elements, zero_elements


