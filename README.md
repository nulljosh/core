# nuLLM - Nano Language Model

**TLDR:** Tiny transformer that learns to predict text, built from scratch in PyTorch.

## What It Does

Learns patterns in text and generates new text based on what it learned.

**Example:**
- Train on Shakespeare → generates Shakespeare-ish text
- Train on code → generates code-ish text
- Train on your messages → generates you-ish text

## How To Use

### 1. Setup
```bash
python3.13 -m venv venv
source venv/bin/activate
pip install torch flask
```

### 1.1 Configuration (Optional)
Create a `.env` file for custom settings:
```bash
MODEL_PATH=models/ultra.pt
DATA_PATH=data/ultra_minimal.txt
PORT=5001
DEBUG=True
```
`.env` is gitignored by default.

### 2. Train
```bash
# Quick test (tiny corpus, 100 epochs)
python3 src/train.py --epochs 100 --corpus tiny

# Better results (more data, more epochs)
python3 src/train.py --epochs 500 --corpus tiny
```

### 3. Generate Text
```bash
# After training, model is saved to models/
python3 src/generate.py --prompt "Whether" --length 50
```

## Architecture

![Architecture](architecture.svg)

**Nano transformer:**
- 4 layers
- 4 attention heads
- 128 embedding dimensions
- ~20K parameters

**Tiny but functional** - learns patterns with minimal compute.

## Training Progress

Loss goes down = model learning:
- Epoch 1: Loss ~3.0 (random garbage)
- Epoch 100: Loss ~0.13 (recognizable patterns)
- Epoch 500: Loss ~0.05 (coherent-ish text)

## What You Can Do With It

1. **Text generation** - Generate creative text
2. **Style transfer** - Train on different writing styles
3. **Code completion** - Train on code, get code suggestions
4. **Chat responses** - Train on conversations
5. **Learn ML** - Understand how transformers work

## Real-World Use

**Current status:** Educational/toy model
- Good for learning transformers
- Good for quick experiments
- NOT production-ready (too small, too simple)

**To make production:**
- Scale up (more layers, bigger embeddings)
- Better data (WikiText-2, OpenWebText)
- Better training (learning rate scheduling, regularization)
- Better sampling (top-k, nucleus, temperature tuning)

## Example Output (100 epochs)

**Prompt:** "Whether"  
**Output:** "Whether tis uer thin the tin..."

Not perfect, but it learned:
- Letter patterns
- Common bigrams (th, er, in)
- Word-ish structures

**With 500 epochs:** More coherent, better word boundaries.

## Files

- `src/train.py` - Training script
- `src/generate.py` - Generation script  
- `src/model.py` - Transformer architecture
- `src/tokenizer.py` - Character-level tokenizer
- `src/data_loader.py` - Dataset loading
- `data/tiny_corpus.txt` - Mini Shakespeare dataset

## Quick Start

```bash
# Train the ultra-minimal model (3 Q&A pairs, fast)
source venv/bin/activate
python3 train_ultra.py  # ~30 seconds

# Start web UI with interactive quiz
python3 web_ui.py
# Visit http://localhost:5001
# Quiz interface: http://localhost:5001/quiz
```

## Web UI

Two interfaces available:

1. **Main UI** (`/`) - Interactive text generation
   - Adjustable temperature, length
   - Live generation

2. **Quiz UI** (`/quiz`) - Test model knowledge
   - 3 questions: name, 5+5, 2+2
   - Shows pass/fail results
   - Clean flat design (no gradients)
   - Progress tracking

## Status

**Latest (Feb 14, 2026):**
- ✅ Ultra-minimal training (3 Q&A pairs, 20K params)
- ✅ Web UI with quiz interface
- ✅ Fixed sequence length bug (IndexError on generation)
- ✅ Clean flat UI design (removed gradients)
- ✅ `.env` configuration support
- ✅ Python 3.13 compatibility

**Model Performance:**
- Math questions (2+2, 5+5): ✅ Working
- Name question: ⚠️ Needs improvement (model too small for longer sequences)

**Next:**
- Scale up model for better name recognition
- Add more training examples
- Implement top-k/nucleus sampling

---

**Built in one night as part of the 3-language sprint**
