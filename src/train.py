"""
Training loop
Simple autoregressive language modeling
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformer import NuLLM
from tokenizer import CharTokenizer
import sys


class TextDataset(Dataset):
    """Simple text dataset for char-level modeling"""
    
    def __init__(self, text, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text)
    
    def __len__(self):
        return len(self.tokens) - self.seq_len
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def train(model, dataloader, optimizer, device, epochs=10):
    """Training loop"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            # Reshape for loss
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")


def generate(model, tokenizer, prompt, max_len=100, device='cpu', temperature=1.0):
    """Generate text from prompt"""
    model.eval()
    
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_len):
            # Get predictions
            logits = model(x)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            
            # Update input
            x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)
            
            # Keep context window reasonable
            if x.size(1) > model.max_len:
                x = x[:, -model.max_len:]
    
    return tokenizer.decode(generated)


if __name__ == "__main__":
    # Sample text (tiny Shakespeare-like)
    text = """
    To be or not to be, that is the question.
    Whether tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles.
    """.strip()
    
    print("Training nuLLM on tiny corpus...")
    print(f"Text length: {len(text)} chars")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = CharTokenizer(text)
    
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # Model config (nano size)
    model = NuLLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        ff_dim=64,
        max_len=64,
        dropout=0.1
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Dataset
    dataset = TextDataset(text, tokenizer, seq_len=32)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train
    print("\nTraining...")
    train(model, dataloader, optimizer, device, epochs=50)
    
    # Generate
    print("\n" + "="*60)
    print("Generating text...")
    print("="*60)
    
    prompts = ["To be", "Whether", "The "]
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        output = generate(model, tokenizer, prompt, max_len=50, device=device, temperature=0.8)
        print(f"Output: {output}")
