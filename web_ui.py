#!/usr/bin/env python3
"""
nuLLM Web UI - Simple Flask interface for text generation
"""

from flask import Flask, render_template, request, jsonify, send_file
import torch
import sys
sys.path.insert(0, 'src')
from transformer import NuLLM
from tokenizer import CharTokenizer
import os

app = Flask(__name__)

# Load model
MODEL_PATH = 'models/ultra.pt'
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
        # Load training data to rebuild tokenizer
        with open('data/ultra_minimal.txt') as f:
            text = f.read()
        tokenizer = CharTokenizer(text)
        
        # Recreate model with same params as training
        model = NuLLM(
            vocab_size=tokenizer.vocab_size,
            embed_dim=32,
            num_heads=2,
            num_layers=2,
            ff_dim=64,
            max_len=32,
            dropout=0.0
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✓ Model loaded from {MODEL_PATH}")
        print(f"✓ Vocab size: {tokenizer.vocab_size} characters")
    else:
        print(f"✗ Model not found at {MODEL_PATH}")
        print("  Train a model first")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/quiz')
def quiz():
    return send_file('quiz.html')

@app.route('/generate', methods=['POST'])
def generate():
    if model is None:
        return jsonify({'error': 'Model not loaded. Train first!'}), 400
    
    data = request.json
    prompt = data.get('prompt', 'The')
    length = int(data.get('length', 100))
    temperature = float(data.get('temperature', 0.8))
    
    try:
        # Encode prompt
        tokens = tokenizer.encode(prompt)
        
        # Trim prompt if too long
        if len(tokens) >= model.max_len:
            tokens = tokens[-(model.max_len-1):]
        
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        generated = tokens.copy()
        
        # Generate tokens
        with torch.no_grad():
            for _ in range(min(length, 50)):  # Cap at 50 tokens
                # Keep sequence under max_len BEFORE forward pass
                if x.size(1) >= model.max_len:
                    x = x[:, -(model.max_len-1):]
                
                logits = model(x)
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.argmax(probs, dim=-1).item()
                generated.append(next_token)
                
                # Stop at newline for Q&A
                decoded = tokenizer.decode(generated)
                if '\n' in decoded[len(prompt):]:
                    break
                
                x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
        
        # Decode
        generated_text = tokenizer.decode(generated)
        return jsonify({'text': generated_text})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    return jsonify({
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'vocab_size': tokenizer.vocab_size if tokenizer else 0
    })

if __name__ == '__main__':
    load_model()
    print("\n🚀 nuLLM Web UI starting...")
    print("   Visit: http://localhost:5001\n")
    app.run(debug=True, port=5001)
