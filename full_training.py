#!/usr/bin/env python3
"""
Enhanced full dataset training script for the neural machine translation model
Uses the complete English-French dataset with optimized hyperparameters and improved translation
"""

import torch
import time
import sys

# Smart tqdm import for both terminal and notebook environments
try:
    if 'ipykernel' in sys.modules or 'IPython' in sys.modules:
        from tqdm.notebook import tqdm
        NOTEBOOK_ENV = True
    else:
        from tqdm import tqdm 
        NOTEBOOK_ENV = False
except ImportError:
    from tqdm import tqdm
    NOTEBOOK_ENV = False

from correct_implementation import train_model, translate_sentence

def train_on_full_dataset():
    """Train the model on progressively larger portions of the dataset"""
    print("=" * 70)
    print("COMPREHENSIVE NEURAL MACHINE TRANSLATION TRAINING")
    print("Training on the full English-French dataset from scratch")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Training configurations for different dataset sizes
    configs = [
        {
            'name': 'Medium Scale Training (25K samples)',
            'sample_size': 25000,
            'epochs': 15,
            'batch_size': 128,
            'embedding_dim': 256,
            'lstm_units': 256,
            'learning_rate': 0.001
        },
        {
            'name': 'Large Scale Training (50K samples)',
            'sample_size': 50000,
            'epochs': 12,
            'batch_size': 256,
            'embedding_dim': 384,
            'lstm_units': 384,
            'learning_rate': 0.0008
        },
        {
            'name': 'Full Dataset Training (All 175K+ samples)',
            'sample_size': None,  # Use all data
            'epochs': 10,
            'batch_size': 512,
            'embedding_dim': 512,
            'lstm_units': 512,
            'learning_rate': 0.0005
        }
    ]
    
    models = {}
    
    # Train on each configuration
    for i, config in enumerate(configs):
        print(f"\n{i+1}. {config['name']}")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            model, data_dict, history = train_model(
                data_file_path='/home/rohan/ws/dltest/dl/eng_-french.csv',
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                embedding_dim=config['embedding_dim'],
                lstm_units=config['lstm_units'],
                learning_rate=config['learning_rate'],
                device=device,
                sample_size=config['sample_size']
            )
            
            training_time = time.time() - start_time
            print(f"\nTraining completed in {training_time:.1f} seconds")
            
            # Save the model configuration and results
            models[config['name']] = {
                'model': model,
                'data_dict': data_dict,
                'history': history,
                'config': config,
                'training_time': training_time
            }
            
            # Test translation quality
            print("\nTesting translation quality:")
            test_sentences = [
                "hello",
                "good morning", 
                "how are you",
                "thank you",
                "what is your name",
                "I love you",
                "goodbye"
            ]
            
            print("=" * 40)
            for sentence in test_sentences[:5]:  # Test first 5 sentences
                try:
                    translation = translate_sentence(
                        model=model,
                        sentence=sentence,
                        eng_tokenizer=data_dict['eng_tokenizer'],
                        fre_tokenizer=data_dict['fre_tokenizer'],
                        max_eng_length=data_dict['max_eng_length'],
                        device=device
                    )
                    print(f"EN: {sentence:15} → FR: {translation}")
                except Exception as e:
                    print(f"EN: {sentence:15} → ERROR: {str(e)[:50]}")
            
            # Print training metrics summary
            final_train_loss = history['train_loss'][-1]
            final_train_acc = history['train_acc'][-1]
            final_val_loss = history['val_loss'][-1] 
            final_val_acc = history['val_acc'][-1]
            
            print(f"\nFinal Metrics:")
            print(f"  Training Loss: {final_train_loss:.4f}")
            print(f"  Training Acc:  {final_train_acc:.4f}")
            print(f"  Validation Loss: {final_val_loss:.4f}")
            print(f"  Validation Acc:  {final_val_acc:.4f}")
            
        except Exception as e:
            print(f"Error during training: {e}")
            continue
    
    # Final comparison
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    
    for name, results in models.items():
        config = results['config']
        history = results['history']
        
        print(f"\n{name}:")
        print(f"  Dataset Size: {config['sample_size'] or '175K+'} samples")
        print(f"  Architecture: {config['embedding_dim']}D embedding, {config['lstm_units']} LSTM units")
        print(f"  Training Time: {results['training_time']:.1f} seconds")
        print(f"  Final Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final Accuracy: {history['train_acc'][-1]:.4f}")
        print(f"  Val Accuracy: {history['val_acc'][-1]:.4f}")
    
    return models

def quick_full_training():
    """Quick training on a substantial portion of the dataset"""
    print("QUICK FULL DATASET TRAINING")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Train on 75K samples with good hyperparameters
    print("Training on 75,000 samples...")
    model, data_dict, history = train_model(
        data_file_path='/home/rohan/ws/dltest/dl/eng_-french.csv',
        epochs=25,
        batch_size=128,
        embedding_dim=384,
        lstm_units=384,
        learning_rate=0.0008,
        device=device,
        sample_size=75000
    )
    
    # Extended testing
    print("\n" + "="*60)
    print("COMPREHENSIVE TRANSLATION TESTING")
    print("="*60)
    
    test_cases = [
        # Basic greetings
        "hello",
        "good morning", 
        "good evening",
        "goodbye",
        
        # Common phrases
        "how are you",
        "what is your name",
        "thank you very much",
        "please help me",
        "excuse me",
        "I am sorry",
        
        # Questions
        "where are you from",
        "what time is it",
        "how much does it cost",
        "where is the bathroom",
        
        # Expressions
        "I love you",
        "I don't understand",
        "can you help me",
        "have a nice day",
        "see you later",
        "nice to meet you"
    ]
    
    successful_translations = 0
    
    for sentence in test_cases:
        try:
            translation = translate_sentence(
                model=model,
                sentence=sentence,
                eng_tokenizer=data_dict['eng_tokenizer'],
                fre_tokenizer=data_dict['fre_tokenizer'],
                max_eng_length=data_dict['max_eng_length'],
                device=device
            )
            
            # Basic quality check - avoid repetitive translations
            words = translation.split()
            unique_ratio = len(set(words)) / max(len(words), 1)
            
            if unique_ratio > 0.3 and len(translation.strip()) > 0:  # Reasonable diversity
                successful_translations += 1
                print(f"✓ EN: {sentence:20} → FR: {translation}")
            else:
                print(f"⚠ EN: {sentence:20} → FR: {translation} (repetitive)")
                
        except Exception as e:
            print(f"✗ EN: {sentence:20} → ERROR: {str(e)[:50]}")
    
    success_rate = successful_translations / len(test_cases) * 100
    
    print(f"\nTranslation Quality Assessment:")
    print(f"Successful translations: {successful_translations}/{len(test_cases)} ({success_rate:.1f}%)")
    print(f"Final training accuracy: {history['train_acc'][-1]:.1%}")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.1%}")
    
    return model, data_dict, history

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "comprehensive":
        # Full comprehensive training
        models = train_on_full_dataset()
    else:
        # Quick substantial training
        model, data_dict, history = quick_full_training()