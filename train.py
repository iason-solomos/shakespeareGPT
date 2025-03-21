import torch
from torch import optim
from typing import Dict

class Trainer:
    def __init__(self, config, model, data_processor):
        self.config = config
        self.model = model
        self.data_processor = data_processor
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        torch.manual_seed(config.seed)

    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        self.model.eval()
        losses = {}
        for split in ['train', 'val']:
            losses[split] = 0
            for _ in range(self.config.eval_iters):
                X, Y = self.data_processor.get_batch(split)
                _, loss = self.model(X, Y)
                losses[split] += loss.item()
            losses[split] /= self.config.eval_iters
        self.model.train()
        return losses

    def train(self):
        for iter in range(self.config.max_iters):
            if iter % self.config.eval_interval == 0 or iter == self.config.max_iters - 1:
                losses = self.estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = self.data_processor.get_batch('train')
            _, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()