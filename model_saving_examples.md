# Model Saving and Loading Examples

This file demonstrates how to use the enhanced model saving functionality.

## Key Features Added

1. **Automatic Best Model Saving**: The training function now automatically saves the model with the highest validation accuracy
2. **Complete State Preservation**: Saves model weights, configuration, data dictionaries, and training history
3. **Easy Loading**: Simple function to load the best saved model for inference or continued training

## Usage Examples

### 1. Training with Automatic Model Saving

```python
# Train a model - it will automatically save the best model based on validation accuracy
model, data_dict, history = train_model_enhanced(
    data_file_path='eng_-french.csv',  # or use_dummy_data=True for testing
    epochs=20,
    batch_size=64,
    embedding_dim=256,
    lstm_units=256,
    encoder_num_layers=2,
    decoder_num_layers=2,
    dropout_rate=0.1,
    bidirectional=True,
    save_path='my_best_model.pt'  # Path where best model will be saved
)
```

### 2. Loading a Saved Model

```python
# Load the best saved model
loaded_model, loaded_data_dict, loaded_history = load_model('my_best_model.pt', device='cuda')

# Use the loaded model for translation
translation = generate("hello world", loaded_model, loaded_data_dict, device='cuda')
print(f"Translation: {translation}")
```

### 3. Continuing Training from Saved Model

```python
# Load a previously saved model
model, data_dict, history = load_model('my_best_model.pt')

# Continue training (note: you'll need to recreate optimizer and scheduler)
# This is useful for fine-tuning or extending training
```

## What Gets Saved

The save function preserves:
- **Model Architecture**: All layer configurations and parameters
- **Model Weights**: Complete state of all trainable parameters
- **Data Processing**: Tokenizers and vocabulary mappings
- **Training History**: Loss and accuracy curves
- **Metadata**: Epoch number, best validation metrics

## Best Practices

1. **Choose Meaningful Paths**: Use descriptive filenames like `nmt_bidirectional_best.pt`
2. **Monitor Training**: The function prints when models are saved
3. **Validation Accuracy Priority**: Models are saved based on validation accuracy, not loss
4. **Device Compatibility**: Models can be loaded on different devices (CPU/GPU)

## Example Output During Training

```
Epoch  5/20 - 15.23s - loss: 2.1432 - acc: 0.3245 - val_loss: 2.0981 - val_acc: 0.3512 - lr: 1.00e-03 - tf: 0.825
Model saved to my_best_model.pt (Epoch 5, Val Acc: 0.3512, Val Loss: 2.0981)

Epoch  8/20 - 15.67s - loss: 1.8923 - acc: 0.4123 - val_loss: 1.9234 - val_acc: 0.4201 - lr: 1.00e-03 - tf: 0.720  
Model saved to my_best_model.pt (Epoch 8, Val Acc: 0.4201, Val Loss: 1.9234)

Training completed!
Best validation accuracy achieved: 0.4201
Best model saved to: my_best_model.pt
```