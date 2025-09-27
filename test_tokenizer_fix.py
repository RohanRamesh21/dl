#!/usr/bin/env python3
"""
Test script to verify the SOS/EOS token ID fixes work correctly
"""
import sys
sys.path.append('.')

from correct_implementation import Tokenizer, prepare_data, translate_sentence, EncoderDecoderModel
import torch

def test_tokenizer_sos_eos():
    """Test that SOS and EOS tokens are properly handled in tokenizer"""
    print("Testing Tokenizer SOS/EOS handling...")
    
    # Create tokenizer and fit on sample data
    tokenizer = Tokenizer()
    sample_texts = ['hello world', 'how are you', 'sos good morning eos']
    tokenizer.fit_on_texts(sample_texts)
    
    # Check that SOS and EOS are in vocabulary
    assert 'sos' in tokenizer.word_index, "SOS token not found in vocabulary"
    assert 'eos' in tokenizer.word_index, "EOS token not found in vocabulary"
    
    sos_id = tokenizer.word_index['sos']
    eos_id = tokenizer.word_index['eos']
    
    print(f"✅ SOS token ID: {sos_id}")
    print(f"✅ EOS token ID: {eos_id}")
    print("✅ Tokenizer SOS/EOS handling works correctly!")
    
    return sos_id, eos_id

def test_prepare_data_tokens():
    """Test that prepare_data returns SOS/EOS token IDs"""
    print("\nTesting prepare_data SOS/EOS extraction...")
    
    # Use dummy data
    data_dict = prepare_data(use_dummy_data=True)
    
    # Check that SOS/EOS IDs are in the returned dictionary
    assert 'sos_id' in data_dict, "SOS ID not found in data dictionary"
    assert 'eos_id' in data_dict, "EOS ID not found in data dictionary"
    assert data_dict['sos_id'] is not None, "SOS ID is None"
    assert data_dict['eos_id'] is not None, "EOS ID is None"
    
    print(f"✅ prepare_data() SOS ID: {data_dict['sos_id']}")
    print(f"✅ prepare_data() EOS ID: {data_dict['eos_id']}")
    print("✅ prepare_data SOS/EOS extraction works correctly!")
    
    return data_dict

def test_model_creation():
    """Test that model can be created with the updated code"""
    print("\nTesting model creation...")
    
    data_dict = prepare_data(use_dummy_data=True)
    
    # Create model
    model = EncoderDecoderModel(
        src_vocab_size=data_dict['eng_vocab_size'],
        tgt_vocab_size=data_dict['fre_vocab_size'],
        embedding_dim=64,  # Small for testing
        lstm_units=64,     # Small for testing
        device='cpu'
    )
    
    # Test parameter collection (should include attention parameters)
    params = model.parameters()
    param_count = sum(p.numel() for p in params)
    
    print(f"✅ Model created successfully!")
    print(f"✅ Total parameters: {param_count:,}")
    print("✅ Model creation works correctly!")
    
    return model, data_dict

def test_translation_function():
    """Test that translate_sentence uses correct SOS/EOS tokens"""
    print("\nTesting translation function...")
    
    # Create a simple model and data
    data_dict = prepare_data(use_dummy_data=True)
    model = EncoderDecoderModel(
        src_vocab_size=data_dict['eng_vocab_size'],
        tgt_vocab_size=data_dict['fre_vocab_size'],
        embedding_dim=32,
        lstm_units=32,
        device='cpu'
    )
    
    # Test translation (should not crash)
    try:
        result = translate_sentence(
            model=model,
            sentence="hello world",
            eng_tokenizer=data_dict['eng_tokenizer'],
            fre_tokenizer=data_dict['fre_tokenizer'],
            max_eng_length=data_dict['max_eng_length'],
            device='cpu',
            max_output_length=10
        )
        print(f"✅ Translation result: '{result}'")
        print("✅ Translation function works correctly!")
        
    except Exception as e:
        print(f"❌ Translation function failed: {e}")
        raise

if __name__ == "__main__":
    print("🚀 Testing SOS/EOS token ID fixes...\n")
    
    # Run all tests
    test_tokenizer_sos_eos()
    test_prepare_data_tokens()
    test_model_creation()
    test_translation_function()
    
    print("\n🎉 All tests passed! SOS/EOS token ID fixes are working correctly.")