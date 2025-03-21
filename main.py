import torch
from config import GPTConfig
from data_processor import DataProcessor
from model import GPTLanguageModel
from train import Trainer

def main():
    config = GPTConfig()
    
    data_processor = DataProcessor(config)
    
    model = GPTLanguageModel(config, data_processor.vocab_size).to(config.device)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Train
    trainer = Trainer(config, model, data_processor)
    trainer.train()
    
    # Generate sample output
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    print(data_processor.decode(model.generate(context, 500)[0].tolist()))

if __name__ == "__main__":
    main()