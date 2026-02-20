from pathlib import Path
import torch
from config import device, block_size, batch_size

# load text
text = Path("aristotle.txt").read_text(encoding="utf-8")

# build vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("vocab_size:", vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]

def decode(indices: list[int]) -> str:
    return "".join(itos[i] for i in indices)

# train/val split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]
print("train tokens:", len(train_data), "val tokens:", len(val_data))

def get_batch(split: str):
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)
