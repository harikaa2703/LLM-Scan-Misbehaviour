import torch
import numpy as np
from scipy.stats import skew, kurtosis
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_layer(model, model_type, layer_idx):
    """
    Helper function to access a specific layer in the model.
    
    Args:
        model: HuggingFace model
        model_type: One of "gpt2", "gptneo", "tinyllama"
        layer_idx: Index of the layer to access
    
    Returns:
        The layer module
    """
    if model_type == "gpt2":
        return model.transformer.h[layer_idx]
    elif model_type == "gptneo":
        return model.transformer.h[layer_idx]
    elif model_type == "tinyllama":
        return model.model.layers[layer_idx]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_num_layers(model, model_type):
    """
    Helper function to get the total number of layers in the model.
    
    Args:
        model: HuggingFace model
        model_type: One of "gpt2", "gptneo", "tinyllama"
    
    Returns:
        Integer count of layers
    """
    if model_type in ("gpt2", "gptneo"):
        return len(model.transformer.h)
    elif model_type == "tinyllama":
        return len(model.model.layers)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def compute_token_causal_effects(model, tokenizer, prompt, model_type):
    """
    Compute the causal effect of each input token using Causal Mediation Analysis (CMA).
    
    For each token in the prompt, replaces it with the intervention token '-', runs a forward pass,
    and computes the L2 distance between original and intervened attention scores.
    
    Args:
        model: A HuggingFace causal language model
        tokenizer: Corresponding tokenizer
        prompt: Input text string
        model_type: One of "gpt2", "gptneo", "tinyllama"
    
    Returns:
        dict with keys:
            - "token_ce": list of float CE values (one per token)
            - "tokens": list of decoded token strings
    """
    model.eval()
    
    # Step 1: Tokenize the prompt
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    
    # Move to model device
    input_ids = input_ids.to(model.device)
    
    # Get token strings for output
    tokens = [tokenizer.decode([token_id.item()]) for token_id in input_ids[0]]
    
    # Determine model architecture, layer count, and which layers/heads to use
    if model_type == "gpt2":
        num_layers = model.config.n_layer
        layers_to_use = [0, 12, 23]
        heads_to_use = [0, 12, 23]
    elif model_type == "gptneo":
        num_layers = model.config.num_layers
        layers_to_use = [0, 6, 11]
        heads_to_use = [0, 6, 11]
    elif model_type == "tinyllama":
        num_layers = model.config.num_hidden_layers
        layers_to_use = [0, 11, 21]
        heads_to_use = [0, 11, 21]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Step 2: Baseline forward pass
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    
    # Extract attention scores from selected layers and heads
    AS_original_arrays = []
    for layer_idx in layers_to_use:
        for head_idx in heads_to_use:
            attn_tensor = outputs.attentions[layer_idx][0, head_idx, :, :].detach().cpu().numpy()
            AS_original_arrays.append(attn_tensor.flatten())
    
    AS_original = np.concatenate(AS_original_arrays)
    
    # Find intervention token id for '-'
    intervention_ids = tokenizer("-", add_special_tokens=False)
    intervention_token_id = intervention_ids.input_ids[0] if intervention_ids.input_ids else tokenizer.unk_token_id
    
    # Step 3: Compute causal effect for each token
    token_ce_list = []
    
    for i in range(input_ids.shape[1]):
        # Create intervened input by replacing token at position i with '-'
        intervened_ids = input_ids.clone()
        intervened_ids[0, i] = intervention_token_id
        
        # Forward pass with intervention
        with torch.no_grad():
            intervened_outputs = model(intervened_ids, output_attentions=True)
        
        # Extract attention scores from selected layers and heads
        AS_intervened_arrays = []
        for layer_idx in layers_to_use:
            for head_idx in heads_to_use:
                attn_tensor = intervened_outputs.attentions[layer_idx][0, head_idx, :, :].detach().cpu().numpy()
                AS_intervened_arrays.append(attn_tensor.flatten())
        
        AS_intervened = np.concatenate(AS_intervened_arrays)
        
        # Compute L2 (Euclidean) distance as causal effect
        ce = np.linalg.norm(AS_original - AS_intervened)
        token_ce_list.append(float(ce))
    
    # Step 4: Return results
    return {
        "token_ce": token_ce_list,
        "tokens": tokens
    }


def extract_token_ce_features(token_ce_list):
    """
    Extract statistical features from token causal effect values.
    
    Args:
        token_ce_list: list of float CE values
    
    Returns:
        numpy array of shape (5,) with [mean, std, range, skewness, kurtosis]
    """
    ce_array = np.array(token_ce_list)
    
    mean_val = np.mean(ce_array)
    std_val = np.std(ce_array)
    range_val = np.max(ce_array) - np.min(ce_array)
    skewness_val = skew(ce_array)
    kurtosis_val = kurtosis(ce_array)
    
    return np.array([mean_val, std_val, range_val, skewness_val, kurtosis_val])


def compute_layer_causal_effects(model, tokenizer, prompt, model_type):
    """
    Compute the causal effect of each transformer layer by skipping it.
    
    For each layer, runs a forward pass where the layer is bypassed (using a hook to 
    replace layer output with layer input), and measures the change in output logits 
    as the causal effect.
    
    Args:
        model: A HuggingFace causal language model
        tokenizer: Corresponding tokenizer
        prompt: Input text string
        model_type: One of "gpt2", "gptneo", "tinyllama"
    
    Returns:
        dict with key:
            - "layer_ce": list of float CE values (one per layer, length = num_layers)
    """
    model.eval()
    
    # Tokenize prompt once
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    
    # Get baseline logits (computed once, reused for all layers)
    with torch.no_grad():
        outputs_baseline = model(input_ids)
    logit_original = outputs_baseline.logits[0, -1, :].detach().cpu().numpy()
    
    layer_ces = []
    num_layers = get_num_layers(model, model_type)
    
    for layer_idx in range(num_layers):
        layer = get_layer(model, model_type, layer_idx)
        
        # Register hook to skip this layer (replace output with input)
        handle = layer.register_forward_hook(lambda m, inp, out: inp[0])
        
        with torch.no_grad():
            outputs_shortcut = model(input_ids)
        
        # Remove hook immediately after shortcut run
        handle.remove()
        
        logit_skipped = outputs_shortcut.logits[0, -1, :].detach().cpu().numpy()
        ce = float(np.linalg.norm(logit_original - logit_skipped))
        layer_ces.append(ce)
    
    return {"layer_ce": layer_ces}


def build_causal_map(model, tokenizer, prompt, model_type):
    """
    Build a comprehensive causal map by combining token and layer causal effects.
    
    Args:
        model: A HuggingFace causal language model
        tokenizer: Corresponding tokenizer
        prompt: Input text string
        model_type: One of "gpt2", "gptneo", "tinyllama"
    
    Returns:
        dict with keys:
            - "tokens": list of token strings
            - "token_ce": list of token causal effects
            - "token_ce_features": list of 5 statistical features
            - "layer_ce": list of layer causal effects
            - "model_type": model type used
            - "prompt": input prompt
    """
    token_result = compute_token_causal_effects(model, tokenizer, prompt, model_type)
    layer_result = compute_layer_causal_effects(model, tokenizer, prompt, model_type)
    
    return {
        "tokens": token_result["tokens"],
        "token_ce": token_result["token_ce"],
        "token_ce_features": extract_token_ce_features(token_result["token_ce"]).tolist(),
        "layer_ce": layer_result["layer_ce"],
        "model_type": model_type,
        "prompt": prompt
    }



if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Test compute_token_causal_effects
    print("=" * 60)
    print("Testing compute_token_causal_effects...")
    print("=" * 60)
    result = compute_token_causal_effects(model, tokenizer, "Who developed Windows 95?", "gpt2")
    print("Tokens:", result["tokens"])
    print("Token CEs:", result["token_ce"])
    print("Features:", extract_token_ce_features(result["token_ce"]))
    
    # Test compute_layer_causal_effects
    print("\n" + "=" * 60)
    print("Testing compute_layer_causal_effects...")
    print("=" * 60)
    layer_result = compute_layer_causal_effects(model, tokenizer, "Who developed Windows 95?", "gpt2")
    layer_ces = layer_result["layer_ce"]
    
    print(f"Number of layers: {len(layer_ces)}")
    print(f"Layer CEs: {layer_ces}")
    
    # Verify all layer CEs are positive
    all_positive = all(ce >= 0 for ce in layer_ces)
    print(f"All layer CEs are non-negative: {all_positive}")
    
    # Verify GPT-2 has 12 layers
    num_layers_gpt2 = get_num_layers(model, "gpt2")
    print(f"GPT-2 has {num_layers_gpt2} layers (expected 12): {num_layers_gpt2 == 12}")
    
    # Compare middle layer (6) vs edge layers (0, 11)
    print(f"\nCausal effect comparison:")
    print(f"  Layer 0 (first):   {layer_ces[0]:.6f}")
    print(f"  Layer 6 (middle):  {layer_ces[6]:.6f}")
    print(f"  Layer 11 (last):   {layer_ces[11]:.6f}")
    middle_larger_than_first = layer_ces[6] > layer_ces[0]
    middle_larger_than_last = layer_ces[6] > layer_ces[11]
    print(f"  Middle > First: {middle_larger_than_first}")
    print(f"  Middle > Last:  {middle_larger_than_last}")
    
    # Test build_causal_map
    print("\n" + "=" * 60)
    print("Testing build_causal_map...")
    print("=" * 60)
    causal_map = build_causal_map(model, tokenizer, "Who developed Windows 95?", "gpt2")
    print(f"Causal map keys: {list(causal_map.keys())}")
    print(f"Model type: {causal_map['model_type']}")
    print(f"Prompt: {causal_map['prompt']}")
    print(f"Number of tokens: {len(causal_map['tokens'])}")
    print(f"Number of layers: {len(causal_map['layer_ce'])}")
    print(f"Token CE features (mean, std, range, skew, kurtosis): {causal_map['token_ce_features']}")

