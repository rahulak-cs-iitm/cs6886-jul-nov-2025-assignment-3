# analyze_sparse_model.py - Analyze sparse model storage breakdown
# This script finds out the sparse weight storage, other parameter storage and metadata storage
import torch
import os
import sys

def analyze_tensor_size(tensor, name="tensor"):
    """
    Calculate actual storage size of a PyTorch tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        return 0, f"Not a tensor: {type(tensor)}"
    
    # Calculate storage size in bytes
    element_size = tensor.element_size()  # bytes per element
    num_elements = tensor.numel()         # total elements
    storage_bytes = element_size * num_elements
    
    info = f"{tensor.shape} {tensor.dtype} = {num_elements:,} elements × {element_size} bytes = {storage_bytes:,} bytes"
    return storage_bytes, info

def analyze_sparse_model(model_path):
    """
    Analyze sparse model to separate weight data from metadata.
    """
    
    if not os.path.exists(model_path):
        print(f"❌ Error: File not found - {model_path}")
        return
    
    print(f"ANALYZING SPARSE MODEL: {model_path}")
    
    # Load the sparse model
    try:
        sparse_data = torch.load(model_path, map_location=device, weights_only=False)
        print(f"Loaded sparse model successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get file size
    total_file_size = os.path.getsize(model_path)
    print(f"Total file size: {total_file_size:,} bytes ({total_file_size/(1024*1024):.2f} MB)")
    print()
    
    # Initialize counters
    weight_data_bytes = 0
    metadata_bytes = 0
    other_params_bytes = 0
    
    print("STORAGE BREAKDOWN:")
    print("-" * 80)
    
    # 1. Analyze sparse weights (actual weight data)
    sparse_weights = sparse_data.get('sparse_weights', {})
    #DEBUG print(f"1. SPARSE WEIGHTS ({len(sparse_weights)} layers):")
    
    for layer_name, layer_data in sparse_weights.items():
        #DEBUG print(f"  📦 {layer_name}:")
        
        # Indices storage
        indices = layer_data.get('indices', [])
        indices_bytes = 0
        for i, idx_tensor in enumerate(indices):
            idx_size, idx_info = analyze_tensor_size(idx_tensor, f"indices_dim{i}")
            indices_bytes += idx_size
            #DEBUG print(f"    Indices dim {i}: {idx_info}")
        
        # Values storage
        values = layer_data.get('values', torch.tensor([]))
        values_bytes, values_info = analyze_tensor_size(values, "values")
        #DEBUG print(f"    Values: {values_info}")
        
        # Shape metadata (small)
        shape = layer_data.get('shape', [])
        shape_bytes = sys.getsizeof(shape)  # Python list overhead
        #DEBUG print(f"    Shape metadata: {shape} = {shape_bytes} bytes")
        
        layer_total = indices_bytes + values_bytes + shape_bytes
        weight_data_bytes += layer_total
        
        #DEBUG print(f"    Layer total: {layer_total:,} bytes")
        #DEBUG print()
    
    print(f"Total sparse weights: {weight_data_bytes:,} bytes ({weight_data_bytes/(1024*1024):.2f} MB)")
    print()
    
    # 2. Analyze other parameters (BatchNorm, biases, etc.)
    other_params = sparse_data.get('other_params', {})
    print(f"2. OTHER PARAMETERS ({len(other_params)} items):")
    
    for param_name, param_tensor in other_params.items():
        param_bytes, param_info = analyze_tensor_size(param_tensor, param_name)
        other_params_bytes += param_bytes
        #DEBUG print(f"  {param_name}: {param_info}")
    
    print(f"Total other params: {other_params_bytes:,} bytes ({other_params_bytes/(1024*1024):.2f} MB)")
    print()
    
    # 3. Analyze scales (for int8 models)
    scales = sparse_data.get('scales', {})
    scales_bytes = 0
    if scales:
        print(f"3. SCALES ({len(scales)} items):")
        for scale_name, scale_value in scales.items():
            scale_size = sys.getsizeof(scale_value)  # Usually float64
            scales_bytes += scale_size
            #DEBUG print(f"  {scale_name}: {scale_value:.8f} = {scale_size} bytes")
        print(f"Total scales: {scales_bytes:,} bytes")
        print()
    
    # 4. Analyze metadata
    metadata = sparse_data.get('metadata', {})
    metadata_bytes = sys.getsizeof(metadata)
    if metadata:
        print(f"4. METADATA:")
        print(f"  {metadata}")
        print(f"  Metadata size: {metadata_bytes} bytes")
        print()
    
    # 5. Calculate and display breakdown
    actual_weight_data = weight_data_bytes  # Sparse indices + values
    actual_other_data = other_params_bytes + scales_bytes  # Non-sparse params
    actual_metadata = metadata_bytes + (len(sparse_weights) * 50)  # Rough estimate for dict overhead
    
    total_accounted = actual_weight_data + actual_other_data + actual_metadata
    unaccounted = total_file_size - total_accounted
    
    print("=" * 80)
    print("Profile of Storage Use:")
    print("=" * 80)
    
    print(f"{'Category':<25} | {'Bytes':<15} | {'MB':<8} | {'%':<8}")
    print("-" * 65)
    
    categories = [
        ("Sparse Weight Data", actual_weight_data),
        ("Other Parameters", actual_other_data), 
        ("Metadata/Overhead", actual_metadata + unaccounted),
        ("TOTAL", total_file_size)
    ]
    
    for category, size in categories:
        mb_size = size / (1024 * 1024)
        percentage = (size / total_file_size) * 100 if total_file_size > 0 else 0
        
        if category == "TOTAL":
            print("-" * 65)
            print(f"{'TOTAL':<25} | {size:<15,} | {mb_size:<8.2f} | {percentage:<8.1f}")
        else:
            print(f"{category:<25} | {size:<15,} | {mb_size:<8.2f} | {percentage:<8.1f}")
    
    return {
        'weight_data_bytes': actual_weight_data,
        'other_params_bytes': actual_other_data,
        'metadata_bytes': actual_metadata + unaccounted,
        'total_bytes': total_file_size
    }

if __name__ == "__main__":
    
    # Model path
    SPARSE_MODEL = "./checkpoints/mobilenetv2_sparse_compression.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if file exists
    if not os.path.exists(SPARSE_MODEL):
        print(f"Error: File not found - {SPARSE_MODEL}")
        exit(1)
    
    # Analyze the sparse model
    breakdown = analyze_sparse_model(SPARSE_MODEL)
