# nuLLM - Claude Notes

## Project Overview
Minimal transformer-based language model. ~500 LOC Python. Educational tool to understand LLM internals.

## Architecture

### Components (src/)
**tokenizer.py**: CharTokenizer (simplest), WordTokenizer (whitespace split), BPE (subword, industry standard)
**attention.py**: SelfAttention (single head, Q/K/V projections), MultiHeadAttention (parallel attention patterns)
**transformer.py**: FeedForward (position-wise FFN), TransformerBlock (attention + FFN + LayerNorm + residuals), NuLLM (full model)
**train.py**: TextDataset (autoregressive sliding window), training loop (cross-entropy + Adam), generation (sampling)
**chat.py**: Conversational wrapper with auto-training fallback

### Model Configs
**Nano**: 50K params, 2 layers, 2 heads, 32 embed, 64 context (demo)
**Micro**: 500K params, 4 layers, 4 heads, 128 embed, 256 context (learning)
**Mini**: 5M params, 6 layers, 8 heads, 256 embed, 512 context (usable)

## Development

### Setup
```bash
cd ~/Documents/Code/nuLLM
python3 -m venv venv
source venv/bin/activate
pip install torch numpy tiktoken tqdm
```

### Testing
```bash
python examples/quick_test.py      # End-to-end verification
python examples/test_tokenizer.py  # Tokenizer tests
python examples/test_model.py      # Model forward pass
python src/chat.py                 # Chat interface
```

### Training Details
- **Loss**: Cross-entropy on next token prediction
- **Optimizer**: Adam with default params
- **Data**: Autoregressive sliding window (input: tokens[i:i+seq_len], target: tokens[i+1:i+seq_len+1])
- **Generation**: Autoregressive sampling, temperature for randomness
- **Performance**: Nano model trains in ~1 min on M-series Mac, <10ms inference per token

## Key Concepts

### Attention Mechanism
- Q, K, V: Each token gets query, key, value vectors
- Scores: Q·K measures similarity between tokens
- Weights: Softmax of scores = attention weights
- Output: Weighted sum of values
- Multi-head: Different heads learn different patterns (syntax vs semantics)

### Architectural Choices
- **Residual connections**: x = x + f(x) helps gradients flow, allows deeper networks
- **Layer normalization**: Normalizes activations, stabilizes training, applied before attention/FFN (pre-norm)
- **Positional encoding**: Learned embeddings encode position (transformers have no inherent position info)

### Temperature
- Low (0.1): Deterministic, repetitive
- Medium (0.8): Balanced
- High (2.0): Creative, chaotic

## Code Structure
```
nuLLM/
├── src/              # Core implementation
│   ├── tokenizer.py  # Text → tokens
│   ├── attention.py  # Attention mechanisms
│   ├── transformer.py # Full model
│   ├── train.py      # Training + generation
│   └── chat.py       # Chat interface
├── examples/         # Test scripts
│   ├── quick_test.py
│   └── test_model.py
├── data/            # Training corpus
├── models/          # Saved checkpoints
└── venv/            # Virtual environment
```

## Status & Next Steps

### Complete (Production Ready)
- All 6 phases implemented and tested
- Tokenization: char, word, BPE working
- Attention: multi-head with positional encoding
- Transformer: full stack with residuals
- Training: loss converges, backprop functional
- Generation: autoregressive sampling working
- Chat: conversational interface complete

### Future Enhancements (Optional)
1. Train on larger corpus (WikiText-2, Shakespeare)
2. Add sampling strategies (top-k, nucleus)
3. Scale to micro/mini sizes
4. Fine-tuning on specific tasks
5. Automated tests (pytest)
6. Web interface (Flask/FastAPI)
7. Inference optimization (quantization, ONNX)

## References

### Papers
- Attention Is All You Need (Vaswani et al., 2017)
- GPT-2 (Radford et al., 2019)
- BERT (Devlin et al., 2018)

### Code
- nanoGPT (Karpathy) - tiny GPT implementation
- minGPT - minimal GPT in PyTorch
- micrograd - tiny autograd engine

## Comparison to Real Models

| Component | GPT-2 | nuLLM (nano) |
|-----------|-------|--------------|
| Layers | 12 | 2 |
| Heads | 12 | 2 |
| Embed | 768 | 32 |
| Context | 1024 | 64 |
| Params | 124M | 50K |

nuLLM is ~2500x smaller but uses the same architecture. Not replacing Claude - understanding how Claude works.
