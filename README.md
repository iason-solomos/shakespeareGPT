# ShakespeareGPT: A Refactored Version of Karpathy's GPT From Scratch

This project is a **modular, refactored version** of the GPT language model implementation from [Andrej Karpathy's YouTube tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY) titled **"Let's build GPT: from scratch, in code, spelled out."**

> ⚠️ **All credit** for the original model logic, hyperparameters, and training flow goes to **[Andrej Karpathy](https://github.com/karpathy)**.  
> This repo simply reorganizes that code into a cleaner, more modular structure using common software engineering practices like class-based design, separation of concerns, and configuration modules.



## 🔥 Features

- Transformer-based architecture (GPT-like)
- Self-attention, multi-head attention, and feedforward blocks
- Token and positional embeddings
- Training loop with validation
- Text generation capability
- Modular and clean codebase using software engineering best practices

## 🧠 Model Architecture

- Embedding size: 384
- Layers: 6
- Attention heads: 6
- Dropout: 0.2
- Context size: 256 tokens

## 🗂 Project Structure

shakespreareGPT/
├── config.py # Hyperparameters and settings (as a dataclass or config object) 

├── data_processor_.py # Loads and encodes the training text 

├── model.py # Transformer blocks and GPT architecture 

├── train.py # Training loop, optimizer, evaluation 

├── main.py # Entry point to train and sample from the model 

└── input.txt # Raw training data (Tiny Shakespeare)


## 🚀 Getting Started

### 1. Clone the repo


git clone https://github.com/iason-solomos/shakespeareGPT.git

cd shakespeareGPT

### 2. Install dependencies
pip install torch

### 3. Download dataset
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

### 3. Train the model
python main.py
