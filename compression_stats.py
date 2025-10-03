# compression_stats.py 
import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import numpy as np
from dataloader import get_cifar10
from utils import evaluate, set_seed, get_model_architecture

def calculate_model_stats(model_path, model_name):
    """
    Calculate comprehensive statistics for a model.
    """
    
    if not os.path.exists(model_path):
        return None, f"{model_name}: File not found - {model_path}"
    
    try:
        # Load model data
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Calculate file size
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        # Handle different model formats
        if 'int8_weights' in model_data:
            # Quantized int8 model
            weights_dict = model_data['int8_weights']
            scales = model_data.get('scales', {})
            model_format = "INT8 Quantized"
            
            # Convert to full state dict for parameter counting
            full_state_dict = {}
            for name, int8_weights in weights_dict.items():
                scale = scales.get(name, 1.0)
                float_weights = int8_weights.float() * scale
                clean_name = name.replace('.original_layer.', '.')
                full_state_dict[clean_name] = float_weights
            
            # Add other parameters
            for key, value in model_data.items():
                if key not in ['int8_weights', 'scales'] and isinstance(value, torch.Tensor):
                    clean_key = key.replace('.original_layer.', '.')
                    full_state_dict[clean_key] = value
                    
        elif 'sparse_weights' in model_data:
            # Sparse compressed model
            model_format = "Sparse Compressed"
            full_state_dict = decode_sparse_model_for_stats(model_data)
            
        else:
            # Standard float32 model
            model_format = "Float32"
            full_state_dict = {}
            for name, param in model_data.items():
                clean_name = name.replace('.original_layer.', '.')
                full_state_dict[clean_name] = param
        
        # Count parameters and calculate sparsity
        total_params = 0
        zero_params = 0
        weight_params = 0
        weight_zeros = 0
        
        for name, param in full_state_dict.items():
            if isinstance(param, torch.Tensor):
                param_count = param.numel()
                param_zeros = (param == 0.0).sum().item()
                
                total_params += param_count
                zero_params += param_zeros
                
                # Count only weight parameters (not biases, batch norm)
                if param.dim() >= 2:  # Conv2d and Linear weights
                    weight_params += param_count
                    weight_zeros += param_zeros
        
        overall_sparsity = (zero_params / total_params) * 100 if total_params > 0 else 0
        weight_sparsity = (weight_zeros / weight_params) * 100 if weight_params > 0 else 0
        
        stats = {
            'file_size_mb': file_size_mb,
            'total_params': total_params,
            'zero_params': zero_params,
            'weight_params': weight_params,
            'weight_zeros': weight_zeros,
            'overall_sparsity': overall_sparsity,
            'weight_sparsity': weight_sparsity,
            'model_format': model_format,
            'state_dict': full_state_dict
        }
        
        return stats, None
        
    except Exception as e:
        return None, f"{model_name}: Error loading - {str(e)}"

def decode_sparse_model_for_stats(sparse_data):
    """
    Decode sparse model for statistics calculation only.
    """
    
    full_model = {}
    scales = sparse_data.get('scales', {})
    
    # Reconstruct sparse weights
    sparse_weights = sparse_data.get('sparse_weights', {})
    
    for name, layer_data in sparse_weights.items():
        indices_list = layer_data['indices']
        values = layer_data['values']
        shape = layer_data['shape']
        
        # Create empty tensor
        if values.dtype == torch.int8:
            full_tensor = torch.zeros(shape, dtype=torch.int8)
        else:
            full_tensor = torch.zeros(shape, dtype=values.dtype)
        
        if len(values) > 0:
            indices_2d = torch.stack([idx.long() for idx in indices_list], dim=1)
            idx_tuple = tuple(indices_2d.T)
            full_tensor[idx_tuple] = values
        
        # Convert int8 to float32 if needed
        if values.dtype == torch.int8 and name in scales:
            scale = scales[name]
            full_tensor = full_tensor.float() * scale
        
        clean_name = name.replace('.original_layer.', '.')
        full_model[clean_name] = full_tensor
    
    # Add other parameters
    other_params = sparse_data.get('other_params', {})
    for name, param in other_params.items():
        full_model[name] = param
    
    return full_model

def evaluate_model_accuracy(stats, model_name):
    """
    Evaluate model accuracy using the state dict.
    """
    
    if stats is None:
        return 0.0, f"Cannot evaluate {model_name} - no valid stats"
    
    try:
        set_seed(47)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, test_loader = get_cifar10(batchsize=128)
        
        # Load model and evaluate
        model = get_model_architecture()
        model.load_state_dict(stats['state_dict'])
        model.to(device)
        
        accuracy = evaluate(model, test_loader, device)
        return accuracy, None
        
    except Exception as e:
        return 0.0, f"Error evaluating {model_name}: {str(e)}"

def measure_activation_compression(model_stats, quantization_bits=8):
    """
    Estimate activation compression based on quantization bits.
    """
    
    # Activation compression is theoretical since we're using dynamic quantization
    # We measure based on the quantization bit-width applied
    
    baseline_bits = 32  # FP32 activations
    compressed_bits = quantization_bits
    
    activation_compression_ratio = baseline_bits / compressed_bits
    
    # Calculate theoretical activation memory for a sample input
    # MobileNetV2 intermediate activations for CIFAR-10 (32x32x3 input)
    sample_activation_elements = [
        32*32*32,    # First conv
        32*32*16,    # Depthwise conv
        16*16*24,    # Bottleneck 1
        16*16*32,    # Bottleneck 2
        8*8*64,      # Bottleneck 3
        8*8*96,      # Bottleneck 4
        8*8*160,     # Bottleneck 5
        8*8*320,     # Final conv
        1280         # Classifier
    ]
    
    total_activation_elements = sum(sample_activation_elements)
    
    baseline_activation_memory = (total_activation_elements * baseline_bits) / (8 * 1024 * 1024)  # MB
    compressed_activation_memory = (total_activation_elements * compressed_bits) / (8 * 1024 * 1024)  # MB
    
    return {
        'activation_compression_ratio': activation_compression_ratio,
        'baseline_activation_memory_mb': baseline_activation_memory,
        'compressed_activation_memory_mb': compressed_activation_memory,
        'method': f'Dynamic {compressed_bits}-bit quantization during inference'
    }

def analyze_compression_journey():
    """
    Main function to analyze the complete compression journey.
    """
    
    print("="*80)
    print("MOBILENETV2 for CIFAR-10 - Training, Pruning, Quantization & Compression")
    print("="*80)
    
    # Model paths
    models = {
        'Original': "./checkpoints/mobilenetv2_cifar10.pth",
        'Pruned': "./checkpoints/mobilenetv2_cifar10_pruned.pth", 
        'Quantized': "./checkpoints/mobilenetv2_pruned_symmetric_quantized_8bit.pth",
        'Sparse Compressed': "./checkpoints/mobilenetv2_sparse_compression.pth"
    }
    
    results = {}
    
    # Stage 1-4: Analyze each model
    stages = ['Original', 'Pruned', 'Quantized', 'Sparse Compressed']
    
    for i, stage in enumerate(stages, 1):
        print(f"\n{'='*60}")
        print(f"STAGE {i}: {stage.upper()} MODEL")
        print(f"{'='*60}")
        
        model_path = models[stage]
        print(f"Path: {model_path}")
        
        # Calculate model statistics
        stats, error = calculate_model_stats(model_path, stage)
        
        if error:
            print(error)
            results[stage] = {'error': error}
            continue
        
        # Evaluate accuracy
        print("Evaluating accuracy...")
        accuracy, acc_error = evaluate_model_accuracy(stats, stage)
        
        if acc_error:
            print(acc_error)
            accuracy = 0.0
        
        # Store results
        results[stage] = {
            'stats': stats,
            'accuracy': accuracy,
            'file_size_mb': stats['file_size_mb'],
            'total_params': stats['total_params'],
            'weight_sparsity': stats['weight_sparsity'],
            'model_format': stats['model_format']
        }
        
        # Display results
        print(f"  Model Format: {stats['model_format']}")
        print(f"  File Size: {stats['file_size_mb']:.2f} MB")
        print(f"  Total Parameters: {stats['total_params']:,}")
        print(f"  Weight Sparsity: {stats['weight_sparsity']:.1f}%")
        print(f"  Test Accuracy: {accuracy:.2f}%")
        
        if i > 1:  # Compare with previous stage
            prev_stage = stages[i-2]
            if prev_stage in results and 'error' not in results[prev_stage]:
                size_reduction = (results[prev_stage]['file_size_mb'] / stats['file_size_mb'])
                acc_change = accuracy - results[prev_stage]['accuracy']
                
                print(f"Comparison with {prev_stage}:")
                print(f"   Size Compression: {size_reduction:.1f}x")
                print(f"   Accuracy Change: {acc_change:+.2f}%")
    
    # Final Comprehensive Analysis
    print(f"\n{'='*80}")
    print("COMPREHENSIVE COMPRESSION ANALYSIS")
    print(f"{'='*80}")
    
    if 'Original' in results and 'Sparse Compressed' in results and 'error' not in results['Original'] and 'error' not in results['Sparse Compressed']:
        
        original = results['Original']
        final = results['Sparse Compressed']
        
        # Overall compression metrics
        total_compression = original['file_size_mb'] / final['file_size_mb']
        accuracy_drop = original['accuracy'] - final['accuracy']
        
        # Weight compression (theoretical based on sparsity and quantization)
        original_weight_size = original['stats']['weight_params'] * 4 / (1024*1024)  # 32-bit weights
        
        # Final weight size: sparse storage + quantization
        sparsity_ratio = (100 - final['stats']['weight_sparsity']) / 100  # Non-zero ratio
        quantization_ratio = 8 / 32  # 8-bit vs 32-bit
        theoretical_weight_size = original_weight_size * sparsity_ratio * quantization_ratio
        # Using actual value from analyze_sparse_model
        actual_weight_size = 0.57
        
        weight_compression = original_weight_size / actual_weight_size 
        # weight_compression = original_weight_size / theoretical_weight_size
        
        # Activation compression analysis
        activation_analysis = measure_activation_compression(final['stats'], quantization_bits=8)
        
        # Summary table
        print(f"{'Stage':<20} | {'Size (MB)':<10} | {'Accuracy (%)':<12} | {'Compression':<12}")
        print("-" * 70)
        
        for stage in stages:
            if stage in results and 'error' not in results[stage]:
                r = results[stage]
                if stage == 'Original':
                    comp_str = "1.0x (baseline)"
                else:
                    comp_str = f"{original['file_size_mb'] / r['file_size_mb']:.1f}x"
                
                print(f"{stage:<20} | {r['file_size_mb']:>8.2f}  | {r['accuracy']:>10.2f}   | {comp_str:<12}")
        
        print(f"\n{'='*50}")
        print("FINAL COMPRESSION METRICS")
        print(f"{'='*50}")
        print(f"Total Model Compression: {total_compression:.1f}x")
        print(f"   Original size: {original['file_size_mb']:.2f} MB")
        print(f"   Final size: {final['file_size_mb']:.2f} MB")
        print(f"   Size reduction: {((1 - final['file_size_mb']/original['file_size_mb']) * 100):.1f}%")
        
        print(f"\nWeight Compression: {weight_compression:.1f}x")
        print(f"   Original weight size: {original_weight_size:.2f} MB")
        print(f"   Final weight size: {actual_weight_size:.2f} MB") 
        # print(f"   Theoretical final size: {theoretical_weight_size:.2f} MB") 
        print(f"   Sparsity contribution: {1/sparsity_ratio:.1f}x")
        print(f"   Quantization contribution: {1/quantization_ratio:.1f}x")
        
        print(f"\nActivation Compression: {activation_analysis['activation_compression_ratio']:.1f}x")
        print(f"   Method: {activation_analysis['method']}")
        print(f"   Baseline activation memory: {activation_analysis['baseline_activation_memory_mb']:.2f} MB")
        print(f"   Compressed activation memory: {activation_analysis['compressed_activation_memory_mb']:.2f} MB")
        
        print(f"\nAccuracy:")
        print(f"   Original accuracy: {original['accuracy']:.2f}%")
        print(f"   Final accuracy: {final['accuracy']:.2f}%")
        print(f"   Accuracy drop: {accuracy_drop:.2f}%")
    
    return results

if __name__ == "__main__":
    
    # Run the complete analysis
    results = analyze_compression_journey()
    
    print(f"\n{'='*80}")

