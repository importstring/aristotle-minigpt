import os
import json
import math
import torch

from config import (
    device,
    block_size,
    batch_size,
    n_embd,
    n_head,
    n_layer,
    learning_rate,
    max_iters,
    eval_interval,
    top_k,
    grad_clip,
    warmup_iters,
    final_lr_mult,
)
from data import get_batch, vocab_size, stoi, decode
from model import TinyGPT


def build_model() -> TinyGPT:
    model = TinyGPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
    ).to(device)
    print("parameters:", sum(p.numel() for p in model.parameters()))
    return model


def estimate_loss(model: TinyGPT):
    model.eval()
    losses = {"train": 0.0, "val": 0.0}
    with torch.no_grad():
        for split in ["train", "val"]:
            split_losses = []
            for _ in range(10):
                xb, yb = get_batch(split)
                _, loss = model(xb, yb)
                split_losses.append(loss.item())
            losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses


def get_lr(it: int) -> float:
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    final_lr = learning_rate * final_lr_mult
    progress = (it - warmup_iters) / max(1, (max_iters - warmup_iters))
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return final_lr + (learning_rate - final_lr) * coeff


def train():
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val = float("inf")
    best_path = "minigpt_aristotle_best.pt"
    patience = 30
    patience_counter = 0

    train_log = {
        "iter": [],
        "train_loss": [],
        "val_loss": [],
        "config": {
            "n_embd": n_embd,
            "n_layer": n_layer,
            "n_head": n_head,
            "block_size": block_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_iters": max_iters,
            "eval_interval": eval_interval,
        },
    }

    for it in range(max_iters):
        lr = get_lr(it)
        for g in optimizer.param_groups:
            g["lr"] = lr

        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if it % eval_interval == 0:
            losses = estimate_loss(model)
            val_loss = losses["val"]
            print(f"iter {it}: train {losses['train']:.4f}, val {val_loss:.4f} (lr={lr:.2e})")

            train_log["iter"].append(it)
            train_log["train_loss"].append(losses["train"])
            train_log["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_path)
                print(f"  ↳ new best val loss, saved to {best_path}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at iter {it} (no val improvement for {patience} evals).")
                break

    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/aristotle_{n_layer}L_{n_embd}d_{max_iters}it.json"
    with open(log_path, "w") as f:
        json.dump(train_log, f)
    print("saved log to", log_path)


if __name__ == "__main__":
    train()
