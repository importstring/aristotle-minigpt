import torch

from config import device, block_size, n_embd, n_head, n_layer, top_k
from data import stoi, decode, vocab_size
from model import TinyGPT


def load_model(checkpoint_path: str = "minigpt_aristotle_best.pt") -> TinyGPT:
    model = TinyGPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
    ).to(device)

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_prompt(model: TinyGPT, prompt: str, temp: float = 0.8, max_new: int = 400) -> str:
    full_prompt = f"Q: {prompt}\nA:"
    idx = torch.tensor([[stoi[c] for c in full_prompt]], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new, temperature=temp, top_k=top_k)
    generated = decode(out[0].tolist())
    print(generated)
    print("\n" + "-" * 80 + "\n")
    return generated


def interactive_chat(model: TinyGPT):
    while True:
        command = input("Ask Aristotle-GPT a question (or 'quit'): \nQ: ")
        if command.strip().lower() in {"q", "quit", "exit"}:
            break
        print(f"Q: {command}")
        run_prompt(model, command, temp=0.8, max_new=755)


if __name__ == "__main__":
    model = load_model("minigpt_aristotle_best.pt")
    interactive_chat(model)
