import torch
from pydantic import BaseModel, field_validator

# Hyperparameters
class GPTConfig(BaseModel):
    batch_size: int = 64
    block_size: int = 256
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters: int = 200
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2
    seed: int = 1337
    dataset_path: str = 'input.txt'

    @field_validator('device')
    def validate_device(cls, v):
        if v == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            return 'cpu'
        return v

    @field_validator('n_head')
    def validate_heads(cls, v, values):
        if 'n_embd' in values and values['n_embd'] % v != 0:
            raise ValueError('n_embd must be divisible by n_head')
        return v