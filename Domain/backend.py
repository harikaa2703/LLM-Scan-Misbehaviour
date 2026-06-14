from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import sys
import os
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Add the Domain/Domain directory to the path to import the scanner
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Domain'))

from tinyllama_scanner import TinyLlamaScanner
from gptneo_scanner import GPTNeoScanner
from copy_of_llmscanlevel2 import GPT2Scanner

app = Flask(__name__)
CORS(app)

# Scanner instances will be created on demand
scanners = {}
scan_cache = {}
evaluation_metrics_cache = None

TASKS = ('toxicity', 'jailbreak', 'lie', 'backdoor')

# AUC is only meaningful when evaluated against labelled validation/test data.
# The saved sklearn models do not persist ROC data, so the backend exposes the
# available validation score as accuracy and leaves AUC unset unless configured.
TINYLLAMA_AUC = {
    # From the local toxicity training artifact.
    'toxicity': 0.8933333333333333,
}

# Metrics published by the Hugging Face detector model cards used by GPT-2,
# GPT-Neo, and Ministral fallbacks:
# - unitary/toxic-bert reports Detoxify original score 0.98636 for the Toxic
#   Comment Classification Challenge, evaluated as mean AUC.
# - ProtectAI/deberta-v3-base-prompt-injection-v2 reports post-training
#   accuracy 95.25%; the card does not publish ROC AUC.
HF_DETECTOR_METRICS = {
    'toxicity': {
        'auc': 0.98636,
        'accuracy': None,
    },
    'jailbreak': {
        'auc': None,
        'accuracy': 0.9525,
    },
}

MODEL_LABELS = {
    'tinyllama': 'TinyLLaMA',
    'mistral': 'Ministral-3B',
    'gpt2': 'GPT-2',
    'gptneo': 'GPT-Neo',
}

def get_scanner(model_name):
    if model_name not in scanners:
        try:
            if model_name == 'tinyllama':
                scanners[model_name] = TinyLlamaScanner()
                print("✓ TinyLlama scanner initialized")
            elif model_name == 'gptneo':
                scanners[model_name] = GPTNeoScanner()
                print("✓ GPT-Neo scanner initialized")
            elif model_name == 'gpt2':
                scanners[model_name] = GPT2Scanner()
                print("✓ GPT-2 scanner initialized")
            elif model_name == 'mistral':
                scanners[model_name] = TinyLlamaScanner(
                    model_name='ministral/Ministral-3b-instruct',
                    fast_mode=True
                )
                print("✓ Ministral-3B scanner initialized")
            else:
                raise ValueError(f"Unknown model: {model_name}")
        except Exception as e:
            print(f"✗ Error initializing {model_name} scanner: {e}")
            return None
    
    return scanners.get(model_name)

def get_local_model_score(model_file):
    model_path = os.path.join(os.path.dirname(__file__), 'Domain', model_file)
    if not os.path.exists(model_path):
        return None
    try:
        model = joblib.load(model_path)
        score = getattr(model, 'best_validation_score_', None)
        return float(score) if score is not None else None
    except Exception as e:
        print(f"Could not read metric metadata from {model_file}: {e}")
        return None

def build_evaluation_metrics():
    global evaluation_metrics_cache
    if evaluation_metrics_cache is not None:
        return evaluation_metrics_cache

    tinyllama_accuracy = {
        'toxicity': get_local_model_score('mlp_toxic.pkl'),
        'jailbreak': get_local_model_score('mlp_jailbreak.pkl'),
        'lie': get_local_model_score('mlp_real.pkl'),
        'backdoor': get_local_model_score('mlp_backdoor.pkl'),
    }

    rows = []
    for model_id, model_label in MODEL_LABELS.items():
        for task in TASKS:
            if model_id == 'tinyllama':
                accuracy = tinyllama_accuracy.get(task)
                auc = TINYLLAMA_AUC.get(task)
            else:
                detector_metrics = HF_DETECTOR_METRICS.get(task, {})
                accuracy = detector_metrics.get('accuracy')
                auc = detector_metrics.get('auc')
            rows.append({
                'model': model_label,
                'task': task,
                'auc': auc,
                'accuracy': accuracy,
            })
    evaluation_metrics_cache = rows
    return rows

def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy types
        return obj.item()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

def get_score(result, key):
    scores = result.get('baseline_scores', {})
    try:
        return float(scores.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0

def select_intervention_response(result):
    interventions = result.get('interventions', [])
    if not interventions:
        return None

    def drift_value(intervention):
        scores = intervention.get('scores', {})
        try:
            return float(scores.get('semantic_drift_score', 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    best = max(interventions, key=drift_value)
    return best.get('new_response')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'LLMSCAN backend ready - supports Ministral-3B, TinyLlama, GPT-2, and GPT-Neo models'})

@app.route('/scan', methods=['POST'])
def scan():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        model = data.get('model', 'mistral')  # default to Mistral
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        # Get the appropriate scanner for the model
        scanner = get_scanner(model)
        if scanner is None:
            return jsonify({'error': f'Failed to initialize {model} scanner'}), 500
        
        print(f"Processing scan for prompt: {prompt[:50]}... with model: {model}")
        
        # Run the scan
        run_safe_unsafe_comparison = model != 'mistral'
        result = scanner.scan(
            prompt,
            run_safe_unsafe_comparison=run_safe_unsafe_comparison
        )
        
        result = convert_to_serializable(result)
        result['normal_response'] = result.get('baseline_response')
        result['misbehavior_response'] = select_intervention_response(result)
        result['score_summary'] = {
            'toxicity': get_score(result, 'toxicity_score'),
            'bias': get_score(result, 'bias_score'),
            'jailbreak': get_score(result, 'jailbreak_score'),
            'hallucination': get_score(result, 'hallucination_score'),
            'misbehaviour': get_score(result, 'misbehaviour_score'),
        }
        result['evaluation_metrics'] = build_evaluation_metrics()
        result['causal_map'] = {
            'image': result.get('plots', {}).get('activation_heatmap')
        }
        scan_cache[(model, prompt)] = result
        
        return jsonify({
            'model': model,
            'success': True,
            'data': result
        })
    except Exception as e:
        print(f"Error during scan: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/models', methods=['GET'])
def models():
    return jsonify({
        'models': [
            {'id': 'mistral', 'name': 'Ministral-3B'},
            {'id': 'tinyllama', 'name': 'TinyLlama-1.1B'},
            {'id': 'gpt2', 'name': 'GPT-2'},
            {'id': 'gptneo', 'name': 'GPT-Neo'}
        ]
    })

@app.route('/causal-map', methods=['POST'])
def causal_map():
    try:
        data = request.json or {}
        prompt = data.get('prompt', '')
        model = data.get('model', 'mistral')
        first_token_only = bool(data.get('first_token_only', False))

        cached = scan_cache.get((model, prompt))
        if cached is None:
            return jsonify({
                'success': False,
                'error': 'Run /scan for this prompt and model before requesting the causal map.'
            }), 404

        plots = cached.get('plots', {})
        image = plots.get('token_confidence') if first_token_only else plots.get('activation_heatmap')
        if not image:
            image = plots.get('activation_heatmap') or plots.get('layer_influence')

        return jsonify({
            'success': True,
            'data': {
                'image': image,
                'mode': 'first_token' if first_token_only else 'full_response',
                'task': data.get('task', 'toxicity')
            }
        })
    except Exception as e:
        print(f"Error during causal map fetch: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify({
        'success': True,
        'data': build_evaluation_metrics()
    })

if __name__ == '__main__':
    print("Starting LLMSCAN Backend Server...")
    print("API running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
