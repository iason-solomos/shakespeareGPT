import torch
from typing import Tuple, Dict

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.text = self._load_data()
        self.vocab, self.stoi, self.itos = self._create_vocab()
        self.vocab_size = len(self.vocab)
        self.train_data, self.val_data = self._split_data()

    def _load_data(self) -> str:
        with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _create_vocab(self) -> Tuple[list, Dict[str, int], Dict[int, str]]:
        chars = sorted(list(set(self.text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return chars, stoi, itos

    def _split_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int(0.9 * len(data))
        return data[:n], data[n:]

    def encode(self, s: str) -> list:
        return [self.stoi[c] for c in s]

    def decode(self, l: list) -> str:
        return ''.join([self.itos[i] for i in l])

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size,))
        x = torch.stack([data[i:i+self.config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.config.block_size+1] for i in ix])
        return x.to(self.config.device), y.to(self.config.device)