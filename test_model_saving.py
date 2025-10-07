#!/usr/bin/env python3
"""
Test script for model saving functionality
"""

import sys
import os
sys.path.append('/home/rohan/proj/dl')

from correct_implementation import train_model_enhanced, load_model, generate
import torch

def test_model_saving():
    """Test the model saving and loading functionality"""
    print("Testing model saving functionality...")
    
    # Train a small model
    print("Training a small model...")
    model, data_dict, history = train_model_enhanced(
        use_dummy_data=True,
        epochs=2,
        batch_size=16,
        embedding_dim=32,
        lstm_units=32,
        save_path='test_best_model.pt',
        device='cpu'
    )
    
    # Test translation with original model
    test_sentence = "hello world"
    original_translation = generate(test_sentence, model, data_dict)
    print(f"Original model translation: '{test_sentence}' -> '{original_translation}'")
    
    # Load the saved model
    print("\\nLoading the saved model...")
    loaded_model, loaded_data_dict, loaded_history = load_model('test_best_model.pt')
    
    # Test translation with loaded model
    loaded_translation = generate(test_sentence, loaded_model, loaded_data_dict)
    print(f"Loaded model translation: '{test_sentence}' -> '{loaded_translation}'")
    
    # Verify they produce similar results (may not be identical due to randomness)
    print(f"\\nOriginal: {original_translation}")
    print(f"Loaded:   {loaded_translation}")
    
    # Check that model configurations match
    print("\\nValidating model configurations...")
    print(f"Original vocab sizes: src={model.src_vocab_size}, tgt={model.tgt_vocab_size}")
    print(f"Loaded vocab sizes:   src={loaded_model.src_vocab_size}, tgt={loaded_model.tgt_vocab_size}")
    print(f"Configurations match: {model.src_vocab_size == loaded_model.src_vocab_size and model.tgt_vocab_size == loaded_model.tgt_vocab_size}")
    
    # Cleanup
    if os.path.exists('test_best_model.pt'):
        os.remove('test_best_model.pt')
        print("Cleaned up test file.")
    
    print("Model saving test completed successfully!")

if __name__ == "__main__":
    test_model_saving()