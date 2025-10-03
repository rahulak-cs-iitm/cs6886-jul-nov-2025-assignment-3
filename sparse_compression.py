# sparse_compression.py
#     - Because theoretical savings aren't enough.
# 
# Pruning and Quantization sparse out the weight storage,
# but actually store the zeros, so model weight doesn't come down
# Sparse compression would help in reducing the model size by 
# storing indices (and values) of non-zero weights
# Notes:
#     - Initial compression took more space (3MB against 2MB) because indices
#       were stored as int64
#     - Updated code to save as int8/16/32 where possible
#     - Finally model came down to ~1MB with negligible accuracy loss from original

import os 
import torch
from dataloader import get_cifar10
from utils import evaluate, set_seed, get_model_architecture

def get_optimal_dtype(max_val):
    """
    Choose the smallest integer type that can hold the maximum index value.
    This was done bcoz by default int64 was used for storing indices
    """
    if max_val < 128:
        return torch.int8
    elif max_val < 32768:
        return torch.int16
    elif max_val < 2147483648:
        return torch.int32
    else:
        return torch.int64

def encode_sparse_model(input_path, output_path):
    """
    Encode model to sparse format - only store non-zero weights + indices.
    Assumes zero-preserving quantization.
    """
    
    print(f"Loading model: {input_path}")
    model_data = torch.load(input_path, map_location=device, weights_only=False)
    
    # Handle int8 model format
    if 'int8_weights' in model_data:
        weights_dict = model_data['int8_weights']
        scales = model_data.get('scales', {})
    # for float32 models
    else:
        weights_dict = model_data
        scales = {}
    
    # Initialize sparse model
    sparse_model = {
        'sparse_weights': {},      # Only sparse weight data
        'other_params': {},        # All other parameters (biases, batch norm, etc.)
        'scales': scales,          # Scale factors for int8 models
        'metadata': {
            'total_weights': 0,
            'stored_values': 0
        }
    }
    
    total_weights = 0
    stored_values = 0
    
    #DEBUG print("\nEncoding layers:")
    
    # Process all parameters
    for name, param in weights_dict.items():
        
        if isinstance(param, torch.Tensor) and param.dim() >= 2:
            # Apply sparse compression to weight matrix 
            
            #DEBUG print(f"  Encoding: {name} {param.shape}")
            
            # Find non-zeros (works directly with int8 due to zero-preservation)
            non_zero_mask = (param != 0)
            non_zero_count = non_zero_mask.sum().item()
            total_elements = param.numel()
            sparsity = (1 - non_zero_count / total_elements) * 100
            
            #DEBUG print(f"    Sparsity: {sparsity:.1f}% ({non_zero_count:,}/{total_elements:,} stored)")
            
            if non_zero_count > 0:
                # Get indices and values
                indices = torch.nonzero(non_zero_mask, as_tuple=False)  # [N, ndims]
                values = param[non_zero_mask]  # [N]
                
                # Optimize index storage - use smallest possible int type per dimension
                optimized_indices = []
                for dim in range(len(param.shape)):
                    max_idx = param.shape[dim] - 1
                    optimal_dtype = get_optimal_dtype(max_idx)
                    dim_indices = indices[:, dim].to(optimal_dtype)
                    optimized_indices.append(dim_indices)
                    #DEBUG print(f"      Dim {dim}: max_idx={max_idx} → {optimal_dtype}")
                
                # Store sparse representation
                sparse_model['sparse_weights'][name] = {
                    'indices': optimized_indices,  # List of optimized index arrays
                    'values': values,              # Non-zero values (keep original dtype)
                    'shape': list(param.shape)    # Original shape
                }
                
                stored_values += non_zero_count
            else:
                print(f"    All zeros - skipping")
            
            total_weights += total_elements
            
        else:
            # Store other parameters (biases, batch norm) directly
            clean_name = name.replace('.original_layer.', '.')
            sparse_model['other_params'][clean_name] = param
            print(f"  Storing: {clean_name} {param.shape if hasattr(param, 'shape') else type(param)}")
    
    # Add any remaining parameters from original model
    if 'int8_weights' in model_data:
        for key, value in model_data.items():
            if key not in ['int8_weights', 'scales'] and isinstance(value, torch.Tensor):
                clean_key = key.replace('.original_layer.', '.')
                sparse_model['other_params'][clean_key] = value
    
    # Store metadata
    sparse_model['metadata']['total_weights'] = total_weights
    sparse_model['metadata']['stored_values'] = stored_values
    overall_sparsity = (1 - stored_values / total_weights) * 100 if total_weights > 0 else 0
    
    # Save sparse model
    torch.save(sparse_model, output_path)
    
    # Calculate compression
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    sparse_size = os.path.getsize(output_path) / (1024 * 1024)
    compression = original_size / sparse_size if sparse_size > 0 else 0
    
    print(f"\n{'='*50}")
    print("ENCODING RESULTS")
    print(f"{'='*50}")
    print(f"Total weights: {total_weights:,}")
    print(f"Stored values: {stored_values:,} ({stored_values/total_weights*100:.1f}%)")
    print(f"Overall sparsity: {overall_sparsity:.1f}%")
    print(f"File size: {original_size:.2f} MB → {sparse_size:.2f} MB")
    print(f"Compression: {compression:.1f}x")
    
    return sparse_model

def decode_sparse_model(sparse_path):
    """
    Decode sparse model back to standard PyTorch format for inference.
    Create and populate full tensor using non-zero's index values
    """
    
    print(f"Decoding sparse model: {sparse_path}")
    sparse_data = torch.load(sparse_path, map_location='cpu', weights_only=False)
    
    full_model = {}
    scales = sparse_data.get('scales', {})
    
    # Reconstruct sparse weights
    sparse_weights = sparse_data.get('sparse_weights', {})
    #DEBUG print(f"Reconstructing {len(sparse_weights)} sparse layers...")
    
    for name, layer_data in sparse_weights.items():
        indices_list = layer_data['indices']  # List of index arrays per dimension
        values = layer_data['values']         # Non-zero values
        shape = layer_data['shape']           # Original shape
        
        #DEBUG print(f"  Reconstructing: {name} {shape}")
        
        # Create empty tensor
        if values.dtype == torch.int8:
            # Int8 model - create int8 tensor, will convert later
            full_tensor = torch.zeros(shape, dtype=torch.int8)
        else:
            full_tensor = torch.zeros(shape, dtype=values.dtype)
        
        if len(values) > 0:
            # Convert list of 1D index arrays back to [N, ndim] format
            indices_2d = torch.stack([idx.long() for idx in indices_list], dim=1)
            
            # Fill tensor using advanced indexing
            idx_tuple = tuple(indices_2d.T)
            full_tensor[idx_tuple] = values
        
        # For int8 models, convert to float32 using scale
        if values.dtype == torch.int8 and name in scales:
            scale = scales[name]
            full_tensor = full_tensor.float() * scale
        
        # Clean parameter name
        clean_name = name.replace('.original_layer.', '.')
        full_model[clean_name] = full_tensor
    
    # Add other parameters
    other_params = sparse_data.get('other_params', {})
    #DEBUG print(f"Adding {len(other_params)} other parameters...")
    
    for name, param in other_params.items():
        full_model[name] = param
    
    return full_model

def evaluate_original_model(input_path):
    """
    Evaluate accuracy of the original (non-compressed) model.
    """
    
    print("Loading original (int8) model...")
    model_data = torch.load(input_path, map_location='cpu', weights_only=False)
    
    # Handle int8 model format
    full_model = {}
    
    if 'int8_weights' in model_data:
        # Int8 model - convert back to float32
        scales = model_data.get('scales', {})
        
        for name, int8_weights in model_data['int8_weights'].items():
            scale = scales.get(name, 1.0)
            float_weights = int8_weights.float() * scale
            clean_name = name.replace('.original_layer.', '.')
            full_model[clean_name] = float_weights
        
        # Add other parameters
        for key, value in model_data.items():
            if key not in ['int8_weights', 'scales'] and isinstance(value, torch.Tensor):
                clean_key = key.replace('.original_layer.', '.')
                full_model[clean_key] = value
    else:
        # Standard float32 model
        for name, param in model_data.items():
            clean_name = name.replace('.original_layer.', '.')
            full_model[clean_name] = param
    
    # Load into model and evaluate
    set_seed(47)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_cifar10(batchsize=128)
    
    model = get_model_architecture()
    model.load_state_dict(full_model)
    model.to(device)
    
    print("Evaluating original model accuracy...")
    accuracy = evaluate(model, test_loader, device)
    
    return accuracy

def evaluate_compressed_model(sparse_path):
    """
    Evaluate accuracy of the compressed sparse model.
    """
    
    print("Loading compressed model...")
    full_state_dict = decode_sparse_model(sparse_path)
    
    # Load into model and evaluate
    set_seed(47)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_cifar10(batchsize=128)
    
    model = get_model_architecture()
    model.load_state_dict(full_state_dict)
    model.to(device)
    
    print("Evaluating compressed model accuracy...")
    accuracy = evaluate(model, test_loader, device)
    
    return accuracy

def compare_models(input_path, sparse_path):
    """
    Compare accuracy of original vs compressed models.
    """
    
    print(f"\n{'='*60}")
    print("MODEL ACCURACY COMPARISON")
    print(f"{'='*60}")
    
    # Evaluate original model
    print("1. Testing ORIGINAL Model...")
    original_accuracy = evaluate_original_model(input_path)
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    
    print("2. Testing COMPRESSED Model...")
    compressed_accuracy = evaluate_compressed_model(sparse_path)
    compressed_size = os.path.getsize(sparse_path) / (1024 * 1024)
    
    # Calculate metrics
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    accuracy_drop = original_accuracy - compressed_accuracy
    
    # Display results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"{'Model':<15} | {'Size (MB)':<10} | {'Accuracy (%)':<12}")
    print("-" * 60)
    print(f"{'Original':<15} | {original_size:>8.2f}  | {original_accuracy:>10.2f}")
    print(f"{'Compressed':<15} | {compressed_size:>8.2f}  | {compressed_accuracy:>10.2f}   ")
    print("-" * 60)
    print(f"Compression Ratio: {compression_ratio:.1f}x")
    print(f"Accuracy Drop: {accuracy_drop:+.2f}%")
    
    return original_accuracy, compressed_accuracy, compression_ratio

# Main execution
if __name__ == "__main__":
    set_seed(47)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    INPUT_MODEL = "./checkpoints/int8_mobilenetv2_pruned_symmetric_quantized_8bit.pth"
    SPARSE_MODEL = "./checkpoints/mobilenetv2_sparse_compression.pth"
    
    # Check input
    if not os.path.exists(INPUT_MODEL):
        print(f"Input model not found: {INPUT_MODEL}")
        exit(1)
    
    print(f"Input model (int8): {INPUT_MODEL} ({os.path.getsize(INPUT_MODEL)/(1024*1024):.2f} MB)")
    
    # Encode to sparse format
    print("="*60)
    print("STEP 1: ENCODING TO SPARSE FORMAT")
    print("="*60)
    
    sparse_data = encode_sparse_model(INPUT_MODEL, SPARSE_MODEL)
    
    # Compare both models
    print("="*60)
    print("STEP 2: COMPARING MODEL ACCURACIES")  
    print("="*60)
    
    original_acc, compressed_acc, compression = compare_models(INPUT_MODEL, SPARSE_MODEL)
    
    # Final summary
    print("="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Original Accuracy   : {original_acc:.2f}%")
    print(f"Compressed Accuracy : {compressed_acc:.2f}%")
    print(f"Compression Ratio   : {compression:.1f}x")
