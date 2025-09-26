#!/usr/bin/env python3
"""
Test script to demonstrate the neural machine translation implementation
with sample English-French data
"""

import pandas as pd
from correct_implementation import train_model, translate_sentence
import torch

def create_sample_data():
    """Create a sample English-French dataset for testing"""
    data = {
        'english': [
            'hello world',
            'how are you',
            'what is your name',
            'good morning',
            'thank you very much',
            'have a nice day',
            'see you later',
            'I am fine',
            'where are you from',
            'what time is it',
            'I love you',
            'good night',
            'how much does it cost',
            'please help me',
            'excuse me',
            'I do not understand',
            'can you help me',
            'where is the bathroom',
            'what are you doing',
            'nice to meet you'
        ],
        'french': [
            'bonjour monde',
            'comment allez vous',
            'quel est votre nom',
            'bon matin',
            'merci beaucoup',
            'bonne journée',
            'à bientôt',
            'je vais bien',
            'd\'où venez vous',
            'quelle heure est il',
            'je t\'aime',
            'bonne nuit',
            'combien ça coûte',
            'aidez moi s\'il vous plaît',
            'excusez moi',
            'je ne comprends pas',
            'pouvez vous m\'aider',
            'où sont les toilettes',
            'que faites vous',
            'ravi de vous rencontrer'
        ]
    }
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv('/home/rohan/ws/dltest/dl/sample_english_french.csv', index=False)
    print(f"Created sample dataset with {len(df)} sentence pairs")
    return '/home/rohan/ws/dltest/dl/sample_english_french.csv'

def main():
    print("=" * 60)
    print("NEURAL MACHINE TRANSLATION - EXTENDED TEST")
    print("From scratch implementation using PyTorch tensors")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create sample data
    data_file = create_sample_data()
    
    # Train model with the sample data
    print("\nTraining with sample English-French data...")
    model, data_dict, history = train_model(
        data_file_path=data_file,
        epochs=20,  # More epochs for better learning
        batch_size=8,  # Small batch for sample data
        embedding_dim=128,
        lstm_units=64,
        learning_rate=0.001,
        device=device,
        sample_size=None  # Use all data
    )
    
    print("\nTraining History:")
    for epoch in range(len(history['train_loss'])):
        print(f"Epoch {epoch+1:2d}: train_loss={history['train_loss'][epoch]:.4f}, "
              f"train_acc={history['train_acc'][epoch]:.4f}, "
              f"val_loss={history['val_loss'][epoch]:.4f}, "
              f"val_acc={history['val_acc'][epoch]:.4f}")
    
    # Test translation with various sentences
    print("\n" + "="*60)
    print("TRANSLATION TESTING")
    print("="*60)
    
    test_sentences = [
        "hello world",
        "how are you",
        "thank you",
        "good morning",
        "what is your name",
        "I love you",
        "have a nice day",
        "where are you from",
        "good night",
        "please help me"
    ]
    
    print("\nTranslating test sentences:")
    print("-" * 40)
    for sentence in test_sentences:
        try:
            translation = translate_sentence(
                model=model,
                sentence=sentence,
                eng_tokenizer=data_dict['eng_tokenizer'],
                fre_tokenizer=data_dict['fre_tokenizer'],
                max_eng_length=data_dict['max_eng_length'],
                device=device
            )
            print(f"EN: {sentence}")
            print(f"FR: {translation}")
            print()
        except Exception as e:
            print(f"Error translating '{sentence}': {e}")
    
    print("✓ Extended testing completed!")
    print("✓ Model successfully trained on sample dataset!")

if __name__ == "__main__":
    main()