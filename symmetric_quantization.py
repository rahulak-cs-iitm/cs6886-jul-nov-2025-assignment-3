# symmetric_quantization.py : Symmetric Linear Quant with Zero preservation for Sparse models
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import os
import copy
import time
import matplotlib.pyplot as plt

from dataloader import get_cifar10
from utils import evaluate, set_seed, get_model_architecture

# Symmetric quantization ensures 0 remains 0
# Uses scale and shift technique from lectures
class SymmetricQuantizer(nn.Module):
    """
    Symmetric Quantization with Zero-Point Preservation for Sparse Models.
    """
    def __init__(self, num_bits=8):
        super().__init__()
        self.num_bits = num_bits
        if self.num_bits < 32:
            # Symmetric quantization range
            # - 2^N to + 2^N-1
            self.q_min = -2**(self.num_bits - 1)
            self.q_max = 2**(self.num_bits - 1) - 1
        else:
            self.q_min = self.q_max = 0  # No quantization for 32-bit
        
    def forward(self, x):
        if self.num_bits >= 32:
            return x  # No quantization for 32-bit
        
        # Find the maximum absolute value (excluding zeros for sparse tensors)
        # creates a mask with non-zero elements marked True
        non_zero_mask = (x != 0.0)
        if not non_zero_mask.any():
            return x  # All zeros, nothing to quantize
        
        # Calculate scale based on non-zero values only
        max_val = x[non_zero_mask].abs().max()
        
        if max_val == 0:
            return x
        
        # Symmetric quantization: scale = max_val / q_max
        scale = max_val / self.q_max
        
        # Quantize: q = round(x / scale)
        x_quantized = torch.round(x / scale)
        
        # Clamp to quantization range
        x_quantized = torch.clamp(x_quantized, self.q_min, self.q_max)
        
        # Dequantize: x_dequant = q * scale
        x_dequantized = x_quantized * scale
        
        # Set earlier zeros to zeros 
        x_dequantized[~non_zero_mask] = 0.0
        # DEBUG print(x_dequantized)
        # DEBUG print(non_zero_mask)
        # DEBUG return
        
        return x_dequantized

class SymmetricQuantizedLayer(nn.Module):
    """
    Wrapper that applies symmetric quantization to weights and activations.
    """
    def __init__(self, original_layer, weight_bits=8, activation_bits=8):
        super().__init__()
        self.original_layer = original_layer
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        
        # Store original weight statistics 
        self.original_sparsity = self._calculate_sparsity(original_layer.weight.data)
        
        # Apply static symmetric quantization to weights
        self._quantize_weights()
        
        # Create activation quantizer (dynamic)
        # Activations are quantized dynamically bcoz they are not known at runtime,
        # and depends on input
        self.activation_quantizer = SymmetricQuantizer(activation_bits)
        
        # Verify sparsity preservation
        new_sparsity = self._calculate_sparsity(self.original_layer.weight.data)
        # DEBUG print(f"Symmetric quantization applied:")
        # DEBUG print(f"  Weight bits: {weight_bits}, Activation bits: {activation_bits}")
        # DEBUG print(f"  Sparsity preserved: {self.original_sparsity:.2f}% -> {new_sparsity:.2f}%")
        
        # Verify zero preservation
        # DEBUG # original_zeros = (torch.zeros_like(original_layer.weight) == 0).sum().item()
        # DEBUG original_zeros = (original_layer.weight == 0).sum().item()
        # DEBUG new_zeros = (self.original_layer.weight.data == 0).sum().item()
        # DEBUG # new_zeros = (self.original_layer.weight == 0).sum().item()
        # DEBUG if original_zeros == new_zeros:
        # DEBUG     print(f"  Zero preservation: Yes ({new_zeros} zeros maintained)")
        # DEBUG else:
        # DEBUG     print(f"  Zero preservation: No {original_zeros} -> {new_zeros}")
    
    def _calculate_sparsity(self, tensor):
        """Calculate sparsity percentage."""
        total_elements = tensor.numel()
        zero_elements = (tensor == 0).sum().item()
        return (zero_elements / total_elements) * 100
    
    def _quantize_weights(self):
        """Apply static symmetric quantization to weights."""
        with torch.no_grad():
            if self.weight_bits < 32:
                weight_quantizer = SymmetricQuantizer(self.weight_bits)
                quantized_weights = weight_quantizer(self.original_layer.weight.data)
                self.original_layer.weight.data = quantized_weights
    
    def forward(self, x):
        """Forward pass with activation quantization."""
        # Apply symmetric quantization to activations
        x_quantized = self.activation_quantizer(x)
        return self.original_layer(x_quantized)

def apply_symmetric_quantization(model, weight_bits=8, activation_bits=8):
    """
    Apply symmetric quantization to all Conv2d and Linear layers.
    """
    quantized_layers_info = []
    
    def replace_layers(module, name=""):
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child_module, (nn.Conv2d, nn.Linear)):
                # DEBUG print(f"\nProcessing layer: {full_name}")
                
                # Store original statistics
                original_sparsity = ((child_module.weight == 0).sum().item() / 
                                   child_module.weight.numel()) * 100
                
                # Create symmetric quantized version
                sym_layer = SymmetricQuantizedLayer(
                    child_module, weight_bits, activation_bits
                )
                setattr(module, child_name, sym_layer)
                
                # Store layer information
                quantized_layers_info.append({
                    'name': full_name,
                    'original_sparsity': original_sparsity,
                    'weight_shape': child_module.weight.shape,
                    'total_params': child_module.weight.numel(),
                    'zero_params': (child_module.weight == 0).sum().item()
                })
                
                # DEBUG print(f"Symmetric quantization applied to: {full_name}")
                
            else:
                replace_layers(child_module, full_name)
    
    replace_layers(model)
    return model, quantized_layers_info

def compare_quantization_methods(model, test_loader, device, bit_levels=[32, 16, 8, 4]):
    """Compare symmetric quantization at different bit levels (32/16/8/4)."""
    results = {}
    
    # Baseline: Original model
    print("\n=== Baseline: Original Pruned Model ===")
    baseline_acc = evaluate(model, test_loader, device)
    results[32] = {'accuracy': baseline_acc, 'method': 'baseline'}
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    
    # Test symmetric quantization at different bit levels
    for bits in bit_levels[1:]:  # Skip 32-bit (baseline)
        print(f"\n=== Symmetric Quantization: {bits}-bit ===")
        print(f"Weights Quantized to {bits}-bit | Activations Quantized to {bits}-bit")
        # Create fresh model copy
        model_copy = copy.deepcopy(model)
        
        # Apply symmetric quantization
        start_time = time.time()
        sq_model, layer_info = apply_symmetric_quantization(
            model_copy, weight_bits=bits, activation_bits=bits
        )
        quantization_time = time.time() - start_time
        
        sq_model.to(device)
        
        # Evaluate accuracy
        start_time = time.time()
        accuracy = evaluate(sq_model, test_loader, device)
        inference_time = time.time() - start_time
        
        results[bits] = {
            'accuracy': accuracy,
            'method': 'symmetric_quantization',
            'quantization_time': quantization_time,
            'inference_time': inference_time,
            'layer_info': layer_info
        }
        
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Quantization time: {quantization_time:.2f}s")
        print(f"Accuracy drop: {baseline_acc - accuracy:.2f}%")
        
        # Save quantized model
        save_path = f"./checkpoints/mobilenetv2_pruned_symmetric_quantized_{bits}bit.pth"
        torch.save(sq_model.state_dict(), save_path)
        print(f"Saved: {save_path}")

    return results

def plot_quantization_comparison(results):
    """Plot accuracy comparison across different quantization bit widths."""
    
    successful_results = {k: v for k, v in results.items() 
                         if v.get('accuracy', 0) > 0}
    
    if len(successful_results) < 2:
        print("Not enough successful results to plot.")
        return
    
    bit_levels = sorted(successful_results.keys())
    accuracies = [successful_results[bits]['accuracy'] for bits in bit_levels]
    
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    colors = ['blue' if bits == 32 else 'red' for bits in bit_levels]
    bars = plt.bar(range(len(bit_levels)), accuracies, color=colors, alpha=0.7)
    
    # Customize plot
    plt.title('Symmetric Quantization: Accuracy vs Bit-Width\n(both weights and activations quantized)')
    plt.xlabel('Quantization Bit-Width')
    plt.ylabel('Test Accuracy (%)')
    plt.xticks(range(len(bit_levels)), [f'{bits}-bit' for bits in bit_levels])
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add legend
    plt.legend(['32-bit (Baseline)', '16/8/4-bit (Symmetric)'])
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('./plots', exist_ok=True)
    save_path = './plots/symmetric_quantization_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved: {save_path}")

def calculate_theoretical_model_sizes(model, results):
    """
    Calculate theoretical model sizes for different quantization bit-widths.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model sizes in MB for different bit-widths
    """
    
    # Count total parameters (weights + biases)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate sizes for different bit-widths
    sizes = {
        '32-bit': (total_params * 32) / (8 * 1024 * 1024),  # 32 bits per param
        '16-bit': (total_params * 16) / (8 * 1024 * 1024),  # 16 bits per param  
        '8-bit':  (total_params * 8) / (8 * 1024 * 1024),   # 8 bits per param
        '4-bit':  (total_params * 4) / (8 * 1024 * 1024)    # 4 bits per param
    }
    
    # Print results
    print(f"\nModel Size Analysis:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  32-bit model size: {sizes['32-bit']:.2f} MB | Accuracy: {results[32]['accuracy']}%")
    print(f"  16-bit model size: {sizes['16-bit']:.2f} MB | Accuracy: {results[16]['accuracy']}%")
    print(f"  8-bit model size:  {sizes['8-bit']:.2f} MB | Accuracy: {results[8]['accuracy']}%")
    print(f"  4-bit model size:  {sizes['4-bit']:.2f} MB | Accuracy: {results[4]['accuracy']}%")
    
    return sizes

if __name__ == "__main__":
    # Configuration
    PRUNED_MODEL_PATH = "./checkpoints/mobilenetv2_cifar10_pruned.pth"
    BIT_LEVELS = [32, 16, 8, 4]
    BATCH_SIZE = 128
    
    # Setup
    set_seed(47)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data and model
    _, test_loader = get_cifar10(batchsize=BATCH_SIZE)
    
    if not os.path.exists(PRUNED_MODEL_PATH):
        print(f"Error: Pruned model not found at {PRUNED_MODEL_PATH}")
        print("Please run iterative_pruning.py first.")
        exit()
    
    print(f"Loading pruned model from: {PRUNED_MODEL_PATH}")
    model = get_model_architecture()
    model.load_state_dict(torch.load(PRUNED_MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)

    print(f"\n{'='*70}")
    print("Starting Symmetric Quantization")
    print(f"{'='*70}")

    results = compare_quantization_methods(model, test_loader, device, BIT_LEVELS)
    
    # Generate comparison plot
    plot_quantization_comparison(results)
    
    # Theoretical size reduction
    sizes = calculate_theoretical_model_sizes(model, results)

