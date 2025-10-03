# gen_int8_model.py
# Currently .pth file saves values as float32, so model file size does not change
# Storing them as int8 would reduce size
import torch
import os
from dataloader import get_cifar10
from utils import evaluate, set_seed, get_model_architecture

def save_model_as_int8(original_model_path, int8_model_path):
  """
  Convert float32 pth file to int8
  """
  print(f"Converting {original_model_path} to int8")

  # Load quantized model
  state_dict = torch.load(original_model_path, map_location=device, weights_only=False)

  int8_model = {
        'int8_weights': {},
        'scales': {},
        'shapes': {}
    }
  int8_min = -2**(8-1)  # -128 for 8-bit
  int8_max = 2**(8-1) - 1 # 127 for 8-bit
  # DEBUG print(f"int8_min = {int8_min} | int8_max = {int8_max}")

  for layer_name, weights in state_dict.items():
    if isinstance(weights, torch.Tensor) and weights.dim() >= 2:
      non_zero_mask = (weights != 0.0)
      if non_zero_mask.any():
        # Calculate scale
        max_val = weights[non_zero_mask].abs().max().item()
        scale = max_val / int8_max

        # Convert to int8
        int8_weights = torch.round(weights/scale).clamp(int8_min,int8_max).to(torch.int8)

        # Store to model
        int8_model['int8_weights'][layer_name] = int8_weights
        int8_model['scales'][layer_name] = scale
        int8_model['shapes'][layer_name] = list(weights.shape)

        # Verify conversion
        reconstructed = int8_weights.float() * scale
        max_error = (reconstructed - weights).abs().max().item()
        # DEBUG print(f"    Max reconstruction error: {max_error:.8f}")
      else:
        # Storing zeros as int8 zeros
        int8_model['int8_weights'][layer_name] = weights.to(torch.int8)
        int8_model['scales'][layer_name] = 1.0
        int8_model['shapes'][layer_name] = list(weights.shape)

    else:
      # dont_touch other parameters like biases
      int8_model[layer_name] = weights

  # Save int8 model
  torch.save(int8_model, int8_model_path)

  # Compare sizes
  original_size = os.path.getsize(original_model_path) / (1024 * 1024)
  int8_size = os.path.getsize(int8_model_path) / (1024 * 1024)
  
  print(f"Original size: {original_size:.2f} MB")
  print(f"Int8 size: {int8_size:.2f} MB")
  print(f"Compression: {original_size/int8_size:.1f}x")


def test_int8_model(int8_model_path):
    # Setup
    set_seed(47)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    _, test_loader = get_cifar10(batchsize=128)
    
    # Load int8 model
    int8_data = torch.load(int8_model_path, map_location=device, weights_only=False)
    
    # Convert int8 back to float32
    # Reconstruct float32 state dict with proper scaling
    clean_state = {}
    
    # Handle int8 weights with scaling
    for layer_name in int8_data['int8_weights']:
        int8_weights = int8_data['int8_weights'][layer_name]
        scale = int8_data['scales'][layer_name]
        
        # Convert back to float32 with proper scaling
        float32_weights = int8_weights.float() * scale
        
        # Update the key name
        # To avoid missing/unexpected key error due to 'original_layer'
        clean_key = layer_name.replace('.original_layer.', '.')
        clean_state[clean_key] = float32_weights
    
    # Handle other parameters (biases, etc.)
    for key, value in int8_data.items():
        if key not in ['int8_weights', 'scales', 'shapes']:
            # To avoid missing/unexpected key error due to 'original_layer'
            clean_key = key.replace('.original_layer.', '.')
            clean_state[clean_key] = value
    
    # Create model and load weights
    model = get_model_architecture()
    model.load_state_dict(clean_state)
    model.to(device)
    
    # Evaluate
    print("Evaluating accuracy...")
    accuracy = evaluate(model, test_loader, device)
    print(f"Int8 model accuracy: {accuracy:.2f}%")
    
if __name__ == "__main__":
    set_seed(47)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    QUANTIZED_MODEL_PATH =  "./checkpoints/mobilenetv2_pruned_symmetric_quantized_8bit.pth"
    INT8_MODEL_PATH = "./checkpoints/int8_mobilenetv2_pruned_symmetric_quantized_8bit.pth"
    save_model_as_int8(QUANTIZED_MODEL_PATH, INT8_MODEL_PATH)
    test_int8_model(INT8_MODEL_PATH)
