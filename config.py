import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# dataset / model config
block_size   = 64    
batch_size   = 32
n_embd       = 192
n_head       = 4
n_layer      = 3      
learning_rate = 3e-4
max_iters     = 15_000  
eval_interval = 200
top_k         = 20
grad_clip     = 1.0

warmup_iters  = 100
final_lr_mult = 0.1   
