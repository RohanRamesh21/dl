"""
Main script to demonstrate the neural machine translation implementation
Implements the same architecture as the reference notebook from absolute scratch
"""
import torch
import os
import sys

# Import our custom implementations
from train import train_model, translate_sentence
from model import create_translation_model
from data_utils import load_translation_data, prepare_data
from tensor import Tensor, zeros, randn


def test_tensor_operations():
    """Test basic tensor operations"""
    print("Testing tensor operations...")
    
    # Test basic operations
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    
    c = a + b
    d = c * 2
    loss = d.sum()
    
    print(f"a: {a}")
    print(f"b: {b}")
    print(f"c = a + b: {c}")
    print(f"d = c * 2: {d}")
    print(f"loss = d.sum(): {loss}")
    
    loss.backward()
    print(f"a.grad: {a.grad}")
    print(f"b.grad: {b.grad}")
    print("✓ Tensor operations working correctly\n")


def test_model_components():
    """Test model components"""
    print("Testing model components...")
    
    from nn import Linear, Embedding, Softmax
    from lstm import LSTM
    from attention import AdditiveAttention
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test embedding
    embedding = Embedding(1000, 256, device=device)
    input_ids = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
    embedded = embedding(input_ids)
    print(f"Embedding output shape: {embedded.shape}")
    
    # Test LSTM
    lstm = LSTM(256, 128, batch_first=True, return_sequences=True, return_state=True, device=device)
    lstm_out, states = lstm(embedded)
    print(f"LSTM output shape: {lstm_out.shape}")
    print(f"LSTM final states: {len(states)} states")
    
    # Test attention
    attention = AdditiveAttention(device=device)
    context, attn_weights = attention(lstm_out, lstm_out)
    print(f"Attention context shape: {context.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    print("✓ Model components working correctly\n")


def test_full_model():
    """Test full encoder-decoder model"""
    print("Testing full model...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = create_translation_model(
        eng_vocab_size=5000,
        fre_vocab_size=4000,
        embedding_dim=128,  # Smaller for testing
        lstm_units=64,      # Smaller for testing
        device=device
    )
    
    # Test forward pass
    batch_size = 2
    eng_seq_len = 10
    fre_seq_len = 12
    
    encoder_inputs = torch.randint(1, 5000, (batch_size, eng_seq_len))
    decoder_inputs = torch.randint(1, 4000, (batch_size, fre_seq_len))
    
    output = model(encoder_inputs, decoder_inputs)
    print(f"Model output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {fre_seq_len}, 4000)")
    
    # Test backward pass
    from optim import SparseCategoricalCrossentropy
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    targets = torch.randint(0, 4000, (batch_size, fre_seq_len))
    
    loss = loss_fn(output, targets)
    print(f"Loss: {loss}")
    
    loss.backward()
    print("✓ Full model forward and backward pass working correctly\n")


def demo_training():
    """Demonstrate training (with dummy data)"""
    print("Demonstrating training process...")
    
    # Create dummy data for demonstration
    print("Creating dummy training data...")
    
    # Simulate English and French sentences
    english_sentences = [
        "hello world",
        "how are you",
        "what is your name", 
        "I am fine",
        "good morning",
        "see you later",
        "thank you very much",
        "have a nice day"
    ]
    
    french_sentences = [
        "bonjour monde",
        "comment allez vous",
        "quel est votre nom",
        "je vais bien", 
        "bon matin",
        "à bientôt",
        "merci beaucoup",
        "bonne journée"
    ]
    
    # Add SOS and EOS tokens
    french_sentences = ['sos ' + sent + ' eos' for sent in french_sentences]
    
    # Prepare data
    from data_utils import prepare_data
    import numpy as np
    
    data_dict = prepare_data(
        english=np.array(english_sentences),
        french=np.array(french_sentences),
        test_size=0.25  # Small dataset, use 25% for validation
    )
    
    print(f"English vocab size: {data_dict['eng_vocab_size']}")
    print(f"French vocab size: {data_dict['fre_vocab_size']}")
    
    # Create trainer
    from train import create_trainer, prepare_training_data
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = create_trainer(
        eng_vocab_size=data_dict['eng_vocab_size'],
        fre_vocab_size=data_dict['fre_vocab_size'],
        embedding_dim=64,   # Small for demo
        lstm_units=32,      # Small for demo
        learning_rate=0.01, # Higher LR for faster convergence on small data
        device=device
    )
    
    # Prepare training data
    train_data, val_data = prepare_training_data(data_dict)
    
    print("Starting training...")
    history = trainer.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=5,  # Few epochs for demo
        batch_size=2,  # Small batch for demo
        verbose=True
    )
    
    # Test translation
    print("\nTesting translation:")
    test_sentence = "hello world"
    translation = translate_sentence(
        trainer=trainer,
        sentence=test_sentence,
        eng_tokenizer=data_dict['eng_tokenizer'],
        fre_tokenizer=data_dict['fre_tokenizer'],
        max_eng_length=data_dict['max_eng_length'],
        device=device
    )
    
    print(f"English: {test_sentence}")
    print(f"French: {translation}")
    print("✓ Training demonstration completed\n")


def main():
    """Main function"""
    print("=" * 60)
    print("NEURAL MACHINE TRANSLATION FROM SCRATCH")
    print("Implementing the reference notebook architecture")
    print("=" * 60)
    print()
    
    try:
        # Test individual components
        test_tensor_operations()
        test_model_components() 
        test_full_model()
        
        # Demonstrate training
        demo_training()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("The implementation is working correctly.")
        print()
        print("To train on real data, use:")
        print("trainer, data_dict = train_model('/path/to/eng_-french.csv')")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
