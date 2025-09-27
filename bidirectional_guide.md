# 🎯 **Bidirectional LSTM Implementation - Testing Notebook**

This notebook demonstrates the bidirectional LSTM enhancement to our Neural Machine Translation model.

## Usage Examples

### 1. Basic Bidirectional Training
```python
from correct_implementation import train_model_enhanced
from config import get_bidirectional_config
import torch

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Get bidirectional configuration
config = get_bidirectional_config()
config.training.device = device
config.print_config()

# Train with bidirectional LSTM
model, data_dict, history = train_model_enhanced(
    data_file_path="eng_-french.csv",  # Your dataset
    epochs=config.training.epochs,
    batch_size=config.training.batch_size,
    embedding_dim=config.model.embedding_dim,
    lstm_units=config.model.lstm_units,
    learning_rate=config.training.learning_rate,
    device=device,
    sample_size=1000,  # Use subset for testing
    bidirectional=True,  # 🎯 Enable bidirectional processing
    encoder_num_layers=config.model.encoder_num_layers,
    decoder_num_layers=config.model.decoder_num_layers,
    dropout_rate=config.model.dropout_rate,
    teacher_forcing_schedule=config.training.teacher_forcing_schedule
)
```

### 2. Compare Unidirectional vs Bidirectional
```python
# Train unidirectional model
print("Training UNIDIRECTIONAL model...")
uni_model, data_dict, uni_history = train_model_enhanced(
    epochs=10,
    bidirectional=False,  # Unidirectional
    sample_size=500,
    device=device
)

# Train bidirectional model
print("Training BIDIRECTIONAL model...")
bi_model, _, bi_history = train_model_enhanced(
    epochs=10,
    bidirectional=True,   # 🎯 Bidirectional
    sample_size=500,
    device=device
)

# Compare results
print("\\nFinal Results Comparison:")
print(f"Unidirectional - Final Val Accuracy: {uni_history['val_acc'][-1]:.4f}")
print(f"Bidirectional  - Final Val Accuracy: {bi_history['val_acc'][-1]:.4f}")
improvement = (bi_history['val_acc'][-1] - uni_history['val_acc'][-1]) * 100
print(f"Improvement: +{improvement:.2f}%")
```

### 3. Test Translation Quality
```python
from correct_implementation import generate

# Test sentences
test_sentences = [
    "hello world",
    "how are you today",
    "what is your name",
    "I love machine learning",
    "the weather is beautiful"
]

print("Translation Comparison:")
print("=" * 60)
for sentence in test_sentences:
    uni_translation = generate(sentence, uni_model, data_dict, device)
    bi_translation = generate(sentence, bi_model, data_dict, device)
    
    print(f"English: {sentence}")
    print(f"Unidirectional: {uni_translation}")
    print(f"Bidirectional:  {bi_translation}")
    print("-" * 40)
```

### 4. Configuration-Based Training
```python
from config import NMTConfig, get_multilayer_bidirectional_config

# Use predefined advanced configuration
config = get_multilayer_bidirectional_config()
config.training.device = device
config.data.sample_size = 1000
config.print_config()

# Train using configuration
model, data_dict, history = train_model_enhanced(
    **{
        'epochs': config.training.epochs,
        'batch_size': config.training.batch_size,
        'embedding_dim': config.model.embedding_dim,
        'lstm_units': config.model.lstm_units,
        'learning_rate': config.training.learning_rate,
        'encoder_num_layers': config.model.encoder_num_layers,
        'decoder_num_layers': config.model.decoder_num_layers,
        'dropout_rate': config.model.dropout_rate,
        'bidirectional': config.model.bidirectional,  # 🎯 From config
        'teacher_forcing_schedule': config.training.teacher_forcing_schedule,
        'device': device,
        'sample_size': config.data.sample_size
    }
)
```

### 5. Custom Configuration Example
```python
# Create custom configuration
custom_config = NMTConfig()

# Model settings
custom_config.model.bidirectional = True      # 🎯 Enable bidirectional
custom_config.model.embedding_dim = 256
custom_config.model.lstm_units = 256
custom_config.model.encoder_num_layers = 2
custom_config.model.dropout_rate = 0.15

# Training settings
custom_config.training.epochs = 25
custom_config.training.batch_size = 16
custom_config.training.learning_rate = 0.0008
custom_config.training.teacher_forcing_schedule = 'exponential'

# Save configuration
custom_config.save_config("my_bidirectional_config.json")

# Load and use
loaded_config = NMTConfig.load_config("my_bidirectional_config.json")
loaded_config.print_config()
```

### 6. Architecture Analysis
```python
# Analyze model parameters
def analyze_model_architecture(model, bidirectional=False):
    total_params = sum(p.numel() for p in model.parameters())
    
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"Model Architecture Analysis ({'Bidirectional' if bidirectional else 'Unidirectional'}):")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Encoder Parameters: {encoder_params:,}")
    print(f"  Decoder Parameters: {decoder_params:,}")
    
    if bidirectional:
        print(f"  Encoder Output Size: {model.encoder.output_size} (2x hidden_size)")
    else:
        print(f"  Encoder Output Size: {model.encoder.lstm_units}")
    
    return total_params

# Analyze both models
uni_params = analyze_model_architecture(uni_model, bidirectional=False)
bi_params = analyze_model_architecture(bi_model, bidirectional=True)

print(f"\\nParameter Increase: {bi_params - uni_params:,} parameters ({(bi_params/uni_params - 1)*100:.1f}% increase)")
```

## Key Features Implemented

### ✅ **Bidirectional LSTM Core**
- Forward and backward LSTM cells
- Proper state concatenation
- Multi-layer bidirectional support

### ✅ **Decoder Compatibility**
- State projection layers for dimensional matching
- Attention mechanism handles variable encoder output sizes
- Output layer automatically adjusts to concatenated context

### ✅ **Configuration System**
- Simple `bidirectional=True/False` parameter
- Predefined configurations for different use cases
- JSON save/load functionality

### ✅ **Training Integration**
- All training functions support bidirectional parameter
- Automatic parameter counting and device handling
- Performance tracking and comparison tools

## Expected Performance Improvements

Based on the bidirectional architecture:
- **10-20% accuracy improvement** for translation tasks
- **Better handling of long sequences** due to bidirectional context
- **Improved attention quality** with richer encoder representations
- **Enhanced semantic understanding** from both directions

## Quick Start Commands

```python
# Import everything
from correct_implementation import *
from config import *
import torch

# Quick bidirectional training
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, data_dict, history = train_model_enhanced(
    epochs=10,
    sample_size=1000,
    bidirectional=True,  # 🎯 This is the key parameter!
    device=device
)

# Test translation
result = generate("hello world", model, data_dict, device)
print(f"Translation: {result}")
```

Ready to train your bidirectional Neural Machine Translation model! 🚀