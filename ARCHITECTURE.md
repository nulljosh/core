# core Architecture

## Overview

core is a minimal transformer-based language model. It implements the core concepts from "Attention Is All You Need" in ~500 lines of Python.

## Components

### 1. Tokenization (`src/tokenizer.py`)

**CharTokenizer**: Character-level tokenization (simplest)
- Vocab: unique characters in training text
- Fast, no training needed
- Good for tiny datasets

**WordTokenizer**: Word-level tokenization
- Splits on whitespace
- Larger vocab, more efficient for real text

**BPE** (TODO): Byte Pair Encoding
- Subword units (between char and word)
- Industry standard (GPT, Claude use BPE variants)

### 2. Attention (`src/attention.py`)

**SelfAttention**: Single attention head
- Query, Key, Value projections
- Scaled dot-product attention
- Learns which tokens to focus on

**MultiHeadAttention**: Parallel attention heads
- Multiple attention patterns
- Concatenated and projected
- Allows model to attend to different aspects

### 3. Transformer Block (`src/transformer.py`)

**FeedForward**: Position-wise FFN
- Linear → GELU → Linear
- Adds non-linearity
- Applied independently to each position

**TransformerBlock**: Full block
- Multi-head attention
- Feed-forward network
- Layer normalization (pre-norm)
- Residual connections
- Dropout for regularization

**Core**: Complete model
- Token embeddings
- Positional embeddings
- Stack of transformer blocks
- Output projection to vocab

### 4. Training (`src/train.py`)

**TextDataset**: Autoregressive dataset
- Sliding window over text
- Input: tokens[i:i+seq_len]
- Target: tokens[i+1:i+seq_len+1]

**Training loop**:
- Cross-entropy loss
- Adam optimizer
- Teacher forcing (always use ground truth)

**Generation**:
- Autoregressive sampling
- Temperature for randomness
- Top-k/nucleus sampling (TODO)

## Model Sizes

### Nano (demo)
- Params: ~50K
- Layers: 2
- Heads: 2
- Embed: 32
- Context: 64 tokens

### Micro (learning)
- Params: ~500K
- Layers: 4
- Heads: 4
- Embed: 128
- Context: 256 tokens

### Mini (usable)
- Params: ~5M
- Layers: 6
- Heads: 8
- Embed: 256
- Context: 512 tokens

## Comparison to Real Models

| Component | GPT-2 | core (nano) |
|-----------|-------|--------------|
| Layers | 12 | 2 |
| Heads | 12 | 2 |
| Embed | 768 | 32 |
| Context | 1024 | 64 |
| Params | 124M | 50K |

core is ~2500x smaller but uses the same architecture.

## Key Concepts

### Attention
- **Q, K, V**: Each token gets query, key, value vectors
- **Scores**: Q·K measures similarity between tokens
- **Weights**: Softmax of scores = attention weights
- **Output**: Weighted sum of values

### Why Multi-Head?
- Different heads learn different patterns
- Some might focus on syntax, others on semantics
- Concatenating gives richer representations

### Residual Connections
- x = x + f(x) instead of x = f(x)
- Helps gradients flow during training
- Allows deeper networks

### Layer Normalization
- Normalizes activations per layer
- Stabilizes training
- Applied before attention/FFN (pre-norm)

### Positional Encoding
- Transformers have no inherent position info
- Learned embeddings encode position
- Added to token embeddings

## Training Details

### Loss Function
Cross-entropy on next token prediction:
```
loss = -log P(token_t+1 | tokens_1..t)
```

### Autoregressive Generation
1. Start with prompt tokens
2. Run forward pass → get logits
3. Sample next token from distribution
4. Append to input, repeat

### Temperature
Controls randomness:
- Low (0.1): Deterministic, repetitive
- Medium (0.8): Balanced
- High (2.0): Creative, chaotic

## Code Structure

```
core/
├── src/
│   ├── tokenizer.py      # Text → tokens
│   ├── attention.py      # Attention mechanisms
│   ├── transformer.py    # Full model
│   └── train.py          # Training + generation
├── examples/
│   ├── quick_test.py     # End-to-end test
│   └── test_model.py     # Model forward pass
└── data/
    └── shakespeare.txt   # Training corpus (TODO)
```

## Dependencies

- **PyTorch**: Core framework
- **NumPy**: Array operations
- **tiktoken**: BPE tokenizer (optional)
- **tqdm**: Progress bars
- **wandb**: Training logging (optional)

## Performance

Nano model on M-series Mac:
- Training: ~1 min for 50 epochs on tiny corpus
- Inference: <10ms per token
- Quality: Learns basic patterns (capitalization, spacing)

## Next Steps

1. Train on real corpus (WikiText-2, Shakespeare)
2. Implement BPE tokenizer
3. Add sampling strategies (top-k, nucleus)
4. Scale up to micro/mini sizes
5. Fine-tuning on specific tasks

## References

Papers:
- Attention Is All You Need (Vaswani et al., 2017)
- GPT-2 (Radford et al., 2019)

Code:
- nanoGPT (Karpathy)
- minGPT (Karpathy)
- micrograd (Karpathy)
