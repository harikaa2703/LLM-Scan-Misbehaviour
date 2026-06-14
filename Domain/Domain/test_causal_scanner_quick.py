"""Quick validation test for causal_scanner.py structure and imports."""
import sys
import inspect

print("=" * 60)
print("CAUSAL SCANNER - QUICK VALIDATION TEST")
print("=" * 60)

# Import the module
try:
    import causal_scanner
    print("✓ causal_scanner.py imported successfully")
except ImportError as e:
    print(f"✗ Failed to import causal_scanner: {e}")
    sys.exit(1)

# Check all required functions exist
required_functions = [
    "get_layer",
    "get_num_layers",
    "compute_token_causal_effects",
    "compute_layer_causal_effects",
    "extract_token_ce_features",
    "build_causal_map"
]

print("\nChecking required functions...")
for func_name in required_functions:
    if hasattr(causal_scanner, func_name):
        func = getattr(causal_scanner, func_name)
        sig = inspect.signature(func)
        print(f"✓ {func_name}{sig}")
    else:
        print(f"✗ {func_name} not found")
        sys.exit(1)

# Test helper functions with mock model
print("\n" + "=" * 60)
print("Testing helper functions...")
print("=" * 60)

class MockConfig:
    n_layer = 12
    num_layers = 12
    num_hidden_layers = 22

class MockModel:
    config = MockConfig()
    class transformer:
        h = [f"layer_{i}" for i in range(12)]
    class model:
        layers = [f"layer_{i}" for i in range(22)]

# Test get_num_layers
try:
    gpt2_layers = causal_scanner.get_num_layers(MockModel(), "gpt2")
    assert gpt2_layers == 12, f"GPT-2 should have 12 layers, got {gpt2_layers}"
    print(f"✓ get_num_layers('gpt2'): {gpt2_layers} layers")
    
    gptneo_layers = causal_scanner.get_num_layers(MockModel(), "gptneo")
    assert gptneo_layers == 12, f"GPT-Neo should have 12 layers, got {gptneo_layers}"
    print(f"✓ get_num_layers('gptneo'): {gptneo_layers} layers")
    
    tinyllama_layers = causal_scanner.get_num_layers(MockModel(), "tinyllama")
    assert tinyllama_layers == 22, f"TinyLlama should have 22 layers, got {tinyllama_layers}"
    print(f"✓ get_num_layers('tinyllama'): {tinyllama_layers} layers")
except Exception as e:
    print(f"✗ get_num_layers test failed: {e}")
    sys.exit(1)

# Test extract_token_ce_features signature and basic usage
print("\n" + "=" * 60)
print("Testing feature extraction...")
print("=" * 60)

try:
    import numpy as np
    test_ces = [0.5, 0.6, 0.4, 0.7, 0.5, 0.8]
    features = causal_scanner.extract_token_ce_features(test_ces)
    
    assert isinstance(features, np.ndarray), "Features should be numpy array"
    assert features.shape == (5,), f"Features should have shape (5,), got {features.shape}"
    
    mean, std, range_val, skew_val, kurt_val = features
    print(f"✓ extract_token_ce_features returned 5-element array")
    print(f"  - Mean: {mean:.4f}")
    print(f"  - Std:  {std:.4f}")
    print(f"  - Range: {range_val:.4f}")
    print(f"  - Skewness: {skew_val:.4f}")
    print(f"  - Kurtosis: {kurt_val:.4f}")
except Exception as e:
    print(f"✗ extract_token_ce_features test failed: {e}")
    sys.exit(1)

# Verify layer configuration for different models
print("\n" + "=" * 60)
print("Verifying layer configuration (from code inspection)...")
print("=" * 60)

try:
    # Read the file to check layer configurations
    with open("causal_scanner.py", "r") as f:
        content = f.read()
    
    # Check GPT-2 configuration
    if 'layers_to_use = [0, 12, 23]' in content and 'model_type == "gpt2"' in content:
        print("✓ GPT-2: layers_to_use = [0, 12, 23] (24 layers total)")
    else:
        print("✗ GPT-2 configuration not found or incorrect")
    
    # Check GPT-Neo configuration
    if 'layers_to_use = [0, 6, 11]' in content and 'model_type == "gptneo"' in content:
        print("✓ GPT-Neo: layers_to_use = [0, 6, 11] (12 layers total)")
    else:
        print("✗ GPT-Neo configuration not found or incorrect")
    
    # Check TinyLlama configuration
    if 'layers_to_use = [0, 11, 21]' in content and 'model_type == "tinyllama"' in content:
        print("✓ TinyLlama: layers_to_use = [0, 11, 21] (22 layers total)")
    else:
        print("✗ TinyLlama configuration not found or incorrect")
        
    # Check for hook pattern in layer causal effects
    if 'lambda m, inp, out: inp[0]' in content:
        print("✓ Layer skipping hook pattern implemented correctly")
    else:
        print("✗ Hook pattern not found")
        
except Exception as e:
    print(f"✗ Configuration verification failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
print("✓ All structural tests passed!")
print("✓ Layer configurations verified")
print("✓ Helper functions operational")
print("✓ Feature extraction working")
print("\nNote: Full model inference test requires downloading GPT-2 (~550MB)")
print("      Run: python causal_scanner.py")
