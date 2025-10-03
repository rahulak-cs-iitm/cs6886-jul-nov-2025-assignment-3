# configurable_compression.py - Configurable wrapper for symmetric_quantization.py
import os
import sys
import torch
import argparse
import time
from dataloader import get_cifar10
from utils import evaluate, set_seed, get_model_architecture

# Import the quantization functions from symmetric_quantization.py
from symmetric_quantization import (
    apply_symmetric_quantization, 
    compare_quantization_methods,
    plot_quantization_comparison,
    calculate_theoretical_model_sizes
)

def calculate_detailed_compression_metrics(model, weight_bits, activation_bits):
    """
    Calculate detailed theoretical compression metrics for the model.
    
    Args:
        model: PyTorch model
        weight_bits: Bit-width for weight quantization
        activation_bits: Bit-width for activation quantization
    
    Returns:
        Dictionary with comprehensive compression metrics
    """
    
    # Analyze model parameters
    total_params = 0
    weight_params = 0
    other_params = 0
    zero_params = 0
    
    layer_details = []
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        param_zeros = (param == 0.0).sum().item()
        
        total_params += param_count
        zero_params += param_zeros
        
        if param.dim() >= 2:  # Weight tensors (Conv2d, Linear)
            weight_params += param_count
            layer_type = "Weight"
            effective_bits = weight_bits
        else:  # Bias and other parameters
            other_params += param_count
            layer_type = "Other"
            effective_bits = 32  # Keep biases in full precision
        
        sparsity = (param_zeros / param_count) * 100 if param_count > 0 else 0
        
        layer_details.append({
            'name': name,
            'type': layer_type,
            'shape': list(param.shape),
            'params': param_count,
            'zeros': param_zeros,
            'sparsity': sparsity,
            'effective_bits': effective_bits
        })
    
    # Calculate theoretical sizes
    baseline_size_bits = total_params * 32  # FP32 baseline
    baseline_size_mb = baseline_size_bits / (8 * 1024 * 1024)
    
    # Quantized size calculation
    weight_size_bits = weight_params * weight_bits
    other_size_bits = other_params * 32  # Keep other params in FP32
    quantized_size_bits = weight_size_bits + other_size_bits
    quantized_size_mb = quantized_size_bits / (8 * 1024 * 1024)
    
    # Sparsity-aware compression (only for weights)
    overall_sparsity = (zero_params / total_params) * 100 if total_params > 0 else 0
    weight_zeros = sum(detail['zeros'] for detail in layer_details if detail['type'] == 'Weight')
    weight_sparsity = (weight_zeros / weight_params) * 100 if weight_params > 0 else 0
    
    # Theoretical sparse compression (simplified: only store non-zero values + indices)
    non_zero_weights = weight_params - weight_zeros
    if weight_sparsity > 0:
        # Sparse storage: values (weight_bits) + indices (assume 16-bit indices)
        sparse_weight_bits = non_zero_weights * weight_bits + non_zero_weights * 16
        sparse_size_bits = sparse_weight_bits + other_size_bits
        sparse_size_mb = sparse_size_bits / (8 * 1024 * 1024)
    else:
        sparse_size_mb = quantized_size_mb
    
    # Calculate compression ratios
    quantization_compression = baseline_size_mb / quantized_size_mb if quantized_size_mb > 0 else 0
    sparse_compression = baseline_size_mb / sparse_size_mb if sparse_size_mb > 0 else 0
    
    # Activation compression (theoretical for inference)
    activation_compression = 32 / activation_bits if activation_bits < 32 else 1.0
    
    return {
        # Model structure
        'total_params': total_params,
        'weight_params': weight_params,
        'other_params': other_params,
        'overall_sparsity': overall_sparsity,
        'weight_sparsity': weight_sparsity,
        
        # Size calculations (MB)
        'baseline_size_mb': baseline_size_mb,
        'quantized_size_mb': quantized_size_mb,
        'sparse_size_mb': sparse_size_mb,
        
        # Compression ratios
        'quantization_compression': quantization_compression,
        'sparse_compression': sparse_compression,
        'activation_compression': activation_compression,
        'total_theoretical_compression': sparse_compression * activation_compression,
        
        # Bit configurations
        'weight_bits': weight_bits,
        'activation_bits': activation_bits,
        
        # Layer details
        'layer_details': layer_details
    }

def run_configurable_compression(
    pruned_model_path, 
    weight_bits, 
    activation_bits,
    save_model=True,
    evaluate_accuracy=True,
    output_dir="./checkpoints/configurable_compression/"
):
    """
    Run symmetric quantization with configurable bit-widths and report theoretical compression.
    """
    
    print(f"{'='*80}")
    print(f"CONFIGURABLE SYMMETRIC QUANTIZATION")
    print(f"{'='*80}")
    print(f"Input model: {pruned_model_path}")
    print(f"Weight quantization: {weight_bits}-bit")
    print(f"Activation quantization: {activation_bits}-bit")
    print(f"{'='*80}")
    
    # Setup
    set_seed(47)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #DEBUG print(f"Using device: {device}")
    
    # Check if input model exists
    if not os.path.exists(pruned_model_path):
        print(f"Error: Pruned model not found at {pruned_model_path}")
        return None
    
    # Load pruned model
    #DEBUG print(f"Loading pruned model...")
    model = get_model_architecture()
    model.load_state_dict(torch.load(pruned_model_path, map_location=device, weights_only=True))
    model.to(device)
    
    # Calculate theoretical compression metrics BEFORE quantization
    #DEBUG print(f"Analyzing model structure and calculating theoretical compression...")
    compression_metrics = calculate_detailed_compression_metrics(model, weight_bits, activation_bits)
    
    # Display theoretical compression analysis
    #DEBUG print(f"\n{'='*60}")
    #DEBUG print("THEORETICAL COMPRESSION ANALYSIS")
    #DEBUG print(f"{'='*60}")
    
    #DEBUG print(f"Model Structure:")
    #DEBUG print(f"   Total parameters: {compression_metrics['total_params']:,}")
    #DEBUG print(f"   Weight parameters: {compression_metrics['weight_params']:,}")
    #DEBUG print(f"   Other parameters: {compression_metrics['other_params']:,}")
    #DEBUG print(f"   Overall sparsity: {compression_metrics['overall_sparsity']:.1f}%")
    #DEBUG print(f"   Weight sparsity: {compression_metrics['weight_sparsity']:.1f}%")
    
    #DEBUG print(f"\nModel Sizes:")
    #DEBUG print(f"   Baseline (FP32): {compression_metrics['baseline_size_mb']:.2f} MB")
    #DEBUG print(f"   Quantized ({weight_bits}-bit weights): {compression_metrics['quantized_size_mb']:.2f} MB")
    #DEBUG print(f"   Sparse + Quantized: {compression_metrics['sparse_size_mb']:.2f} MB")
    
    #DEBUG print(f"\nTheoretical Compression Ratios:")
    #DEBUG print(f"   Weight quantization: {compression_metrics['quantization_compression']:.1f}x")
    #DEBUG print(f"   Sparse + quantization: {compression_metrics['sparse_compression']:.1f}x")
    #DEBUG print(f"   Activation compression: {compression_metrics['activation_compression']:.1f}x")
    #DEBUG print(f"   TOTAL THEORETICAL: {compression_metrics['total_theoretical_compression']:.1f}x")
    
    # Display per-layer breakdown
    #DEBUG print(f"\nPer-Layer Breakdown:")
    #DEBUG print(f"{'Layer':<30} | {'Type':<8} | {'Params':<12} | {'Sparsity':<10} | {'Bits':<6}")
    #DEBUG print("-" * 80)
    
    for detail in compression_metrics['layer_details'][:10]:  # Show first 10 layers
        layer_name = detail['name'].split('.')[-1][:29] if '.' in detail['name'] else detail['name'][:29]
        #DEBUG print(f"{layer_name:<30} | {detail['type']:<8} | {detail['params']:<12,} | {detail['sparsity']:<9.1f}% | {detail['effective_bits']:<6}")
    
    # if len(compression_metrics['layer_details']) > 10:
        #DEBUG print(f"... and {len(compression_metrics['layer_details']) - 10} more layers")
    
    # Load test data for accuracy evaluation
    if evaluate_accuracy:
        #DEBUG print(f"\n{'='*60}")
        #DEBUG print("MODEL ACCURACY EVALUATION")
        #DEBUG print(f"{'='*60}")
        
        #DEBUG print("Loading CIFAR-10 test data...")
        _, test_loader = get_cifar10(batchsize=128)
        
        # Evaluate original model
        #DEBUG print("Evaluating original pruned model...")
        original_accuracy = evaluate(model, test_loader, device)
        #DEBUG print(f"Original accuracy: {original_accuracy:.2f}%")
        
        # Apply symmetric quantization
        #DEBUG print(f"Applying symmetric quantization...")
        start_time = time.time()
        
        # Create a copy for quantization
        import copy
        model_copy = copy.deepcopy(model)
        
        # Apply quantization using your existing function
        quantized_model, layer_info = apply_symmetric_quantization(
            model_copy,
            weight_bits=weight_bits,
            activation_bits=activation_bits
        )
        quantized_model.to(device)
        
        quantization_time = time.time() - start_time
        #DEBUG print(f"Quantization completed in {quantization_time:.2f} seconds")
        
        # Evaluate quantized model
        #DEBUG print("Evaluating quantized model...")
        start_time = time.time()
        quantized_accuracy = evaluate(quantized_model, test_loader, device)
        evaluation_time = time.time() - start_time
        
        accuracy_drop = original_accuracy - quantized_accuracy
        accuracy_retention = (quantized_accuracy / original_accuracy) * 100 if original_accuracy > 0 else 0
        
        #DEBUG print(f"Quantized accuracy: {quantized_accuracy:.2f}%")
        #DEBUG print(f"Accuracy drop: {accuracy_drop:.2f}%")
        #DEBUG print(f"Accuracy retention: {accuracy_retention:.1f}%")
        #DEBUG print(f"Evaluation completed in {evaluation_time:.2f} seconds")
        
        # Save quantized model
        if save_model:
            os.makedirs(output_dir, exist_ok=True)
            
            output_filename = f"mobilenetv2_quantized_w{weight_bits}bit_a{activation_bits}bit.pth"
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"Saving quantized model to: {output_path}")
            torch.save(quantized_model.state_dict(), output_path)
            
            # Save comprehensive report
            print(f"\n{'='*50}")
            print(f"MobileNetV2 CIFAR-10 Compression Report")
            print(f"{'='*50}\n")
            print(f"Configuration:")
            print(f"  Weight bits: {weight_bits}")
            print(f"  Activation bits: {activation_bits}")
            print(f"  Input model: {pruned_model_path}\n")
            print
            print(f"Model Structure:")
            print(f"  Total parameters: {compression_metrics['total_params']:,}")
            print(f"  Weight parameters: {compression_metrics['weight_params']:,}")
            print(f"  Other parameters: {compression_metrics['other_params']:,}")
            print(f"  Overall sparsity: {compression_metrics['overall_sparsity']:.1f}%")
            print(f"  Weight sparsity: {compression_metrics['weight_sparsity']:.1f}%\n")
            print
            print(f"Theoretical Compression:")
            print(f"  Baseline size: {compression_metrics['baseline_size_mb']:.2f} MB")
            print(f"  Quantized size: {compression_metrics['quantized_size_mb']:.2f} MB")
            print(f"  Sparse + Quantized: {compression_metrics['sparse_size_mb']:.2f} MB")
            print(f"  Weight compression: {compression_metrics['quantization_compression']:.1f}x")
            print(f"  Sparse compression: {compression_metrics['sparse_compression']:.1f}x")
            print(f"  Activation compression: {compression_metrics['activation_compression']:.1f}x")
            print(f"  TOTAL COMPRESSION: {compression_metrics['total_theoretical_compression']:.1f}x\n")
            print
            print(f"Accuracy Results:")
            print(f"  Original accuracy: {original_accuracy:.2f}%")
            print(f"  Quantized accuracy: {quantized_accuracy:.2f}%")
            print(f"  Accuracy drop: {accuracy_drop:.2f}%")
            print(f"  Accuracy retention: {accuracy_retention:.1f}%\n")
            print
            print(f"Performance:")
            print(f"  Quantization time: {quantization_time:.2f}s")
            print(f"  Evaluation time: {evaluation_time:.2f}s")
            
        # Update compression metrics with actual results
        compression_metrics.update({
            'original_accuracy': original_accuracy,
            'quantized_accuracy': quantized_accuracy,
            'accuracy_drop': accuracy_drop,
            'accuracy_retention': accuracy_retention,
            'quantization_time': quantization_time,
            'model_path': output_path if save_model else None
        })
    
    return compression_metrics

def main():
    """Main function with command-line interface."""
    
    parser = argparse.ArgumentParser(
        description='Configurable Symmetric Quantization with Theoretical Compression Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard 8-bit quantization
    python configurable_compression.py --weight_bits 8 --activation_bits 8
    
    # Mixed precision
    python configurable_compression.py --weight_bits 16 --activation_bits 8
    
    # Aggressive compression
    python configurable_compression.py --weight_bits 4 --activation_bits 4
    
    # Only theoretical analysis (no accuracy evaluation)
    python configurable_compression.py --weight_bits 8 --activation_bits 8 --no_eval
        """
    )
    
    # Required arguments
    parser.add_argument('--weight_bits', type=int, required=True,
                       choices=[4, 6, 8, 12, 16, 32],
                       help='Bit-width for weight quantization')
    
    parser.add_argument('--activation_bits', type=int, required=True,
                       choices=[4, 6, 8, 12, 16, 32],
                       help='Bit-width for activation quantization')
    
    # Optional arguments
    parser.add_argument('--pruned_model', type=str,
                       default="./checkpoints/mobilenetv2_cifar10_pruned.pth",
                       help='Path to the pruned model')
    
    parser.add_argument('--output_dir', type=str,
                       default="./checkpoints/configurable_compression",
                       help='Output directory for saving results')
    
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save the quantized model')
    
    parser.add_argument('--no_eval', action='store_true',
                       help='Skip accuracy evaluation (faster, theoretical only)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.pruned_model):
        print(f"Error: Pruned model not found at {args.pruned_model}")
        print("Please ensure the pruned model exists or specify correct path with --pruned_model")
        sys.exit(1)
    
    # Run compression analysis
    try:
        results = run_configurable_compression(
            pruned_model_path=args.pruned_model,
            weight_bits=args.weight_bits,
            activation_bits=args.activation_bits,
            save_model=not args.no_save,
            evaluate_accuracy=not args.no_eval,
            output_dir=args.output_dir
        )
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

