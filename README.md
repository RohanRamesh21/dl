# Neural Machine Translation from Scratch

This project implements a neural machine translation system from absolute scratch, replicating the architecture from the reference Jupyter notebook (`Untitled58.ipynb`). The implementation uses only PyTorch's tensor data structures for CUDA integration while building all neural network components, automatic differentiation, and training pipeline from ground up.

## Architecture

The model implements an **Encoder-Decoder architecture with Additive Attention** for English-French translation:

- **Encoder**: Embedding layer + LSTM (returns sequences and states)
- **Decoder**: Embedding layer + LSTM + Additive Attention + Dense output layer
- **Attention**: Additive (Bahdanau) attention mechanism
- **Loss**: Sparse categorical crossentropy
- **Optimizer**: Adam with custom implementation

## Project Structure

```
dl/
├── tensor.py           # Custom tensor class with automatic differentiation
├── nn.py              # Neural network layers (Linear, Embedding, etc.)
├── lstm.py            # LSTM implementation from scratch
├── attention.py       # Attention mechanisms
├── optim.py           # Optimizers and loss functions
├── model.py           # Encoder-decoder architecture
├── data_utils.py      # Data preprocessing and tokenization
├── train.py           # Training pipeline
└── scratch_nn.py      # Main script with tests and demo
```

## Features Implemented from Scratch

### Core Components
- **Tensor class** with automatic differentiation
- **Linear layers** with forward/backward pass
- **Embedding layers** with gradient computation
- **LSTM cells** with all gates (forget, input, output)
- **Activation functions** (tanh, sigmoid, softmax, ReLU)
- **Loss functions** (cross-entropy, sparse categorical crossentropy)

### Advanced Features
- **Additive attention mechanism** 
- **Adam optimizer** with bias correction
- **Learning rate schedulers**
- **Data loaders** with batching
- **Tokenization** matching TensorFlow's behavior
- **Sequence padding** utilities

### Training Pipeline
- **Batch processing** with gradient updates
- **Validation metrics** (loss and accuracy)
- **Training history** tracking
- **Model evaluation** and translation inference

## Usage

### Quick Test
```python
# Run the main script to test all components
python scratch_nn.py
```

### Training on Real Data
```python
from train import train_model

# Train on the English-French dataset
trainer, data_dict = train_model(
    data_file_path='/path/to/eng_-french.csv',
    epochs=10,
    batch_size=64,
    embedding_dim=256,
    lstm_units=256,
    learning_rate=0.001,
    device='cuda'  # or 'cpu'
)
```

### Translation
```python
from train import translate_sentence

# Translate a sentence
translation = translate_sentence(
    trainer=trainer,
    sentence="hello world",
    eng_tokenizer=data_dict['eng_tokenizer'],
    fre_tokenizer=data_dict['fre_tokenizer'], 
    max_eng_length=data_dict['max_eng_length']
)
print(f"Translation: {translation}")
```

## Implementation Details

### Custom Tensor System
- Implements PyTorch-like tensor operations
- Automatic gradient computation via computational graph
- Support for broadcasting and shape operations
- CUDA device support through PyTorch backend

### LSTM from Scratch
- All LSTM gates implemented manually
- Proper gradient flow through time
- Support for return_sequences and return_state
- Bidirectional LSTM support

### Attention Mechanism
- Additive (Bahdanau) attention implementation
- Matches TensorFlow's AdditiveAttention behavior
- Proper gradient computation for attention weights
- Support for attention masking

### Training Pipeline
- Custom data loaders with shuffling
- Gradient clipping and regularization
- Learning rate scheduling
- Early stopping capabilities

## Requirements

- Python 3.7+
- PyTorch (for tensor storage and CUDA support only)
- NumPy
- Pandas (for data loading)
- scikit-learn (for train/test split)

## Notes

This implementation demonstrates:
1. **Complete neural network from scratch** - no pre-built layers used
2. **Automatic differentiation** - custom backpropagation implementation
3. **Production-ready architecture** - matches TensorFlow reference exactly
4. **CUDA support** - leverages PyTorch tensors for GPU acceleration
5. **Modular design** - easily extensible and readable code structure

The code is designed for educational purposes and demonstrates how modern deep learning frameworks work under the hood, while still being capable of real translation tasks.