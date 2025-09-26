"""
Improved training script with scheduled sampling and beam search
Demonstrates the complete training pipeline with all enhancements
"""
from train import train_model, translate_sentence, evaluate_translations, create_trainer, prepare_training_data
from data_utils import load_translation_data, prepare_data
import matplotlib.pyplot as plt
import time


def train_with_improvements(data_file_path='sample_english_french.csv', 
                          epochs=50, 
                          batch_size=64,
                          embedding_dim=256,
                          lstm_units=256, 
                          learning_rate=0.001,
                          device='cpu',
                          sample_size=1000):
    """
    Train model with all improvements: scheduled sampling, teacher forcing curriculum, etc.
    
    Args:
        data_file_path: Path to training data
        epochs: Number of epochs
        batch_size: Batch size  
        embedding_dim: Embedding dimension
        lstm_units: LSTM hidden size
        learning_rate: Learning rate
        device: Training device
        sample_size: Number of samples to use for training
        
    Returns:
        trainer: Trained model
        data_dict: Data dictionary with tokenizers
    """
    
    print("🚀 Starting Improved Neural Machine Translation Training")
    print("=" * 60)
    
    # Load and prepare data
    print("📊 Loading and preprocessing data...")
    start_time = time.time()
    
    english, french = load_translation_data(data_file_path, sample_size=sample_size)
    data_dict = prepare_data(english, french)
    
    print(f"✅ Data loaded in {time.time() - start_time:.2f}s")
    print(f"   📈 English vocabulary: {data_dict['eng_vocab_size']}")
    print(f"   📈 French vocabulary: {data_dict['fre_vocab_size']}")
    print(f"   📏 Max English length: {data_dict['max_eng_length']}")
    print(f"   📏 Max French length: {data_dict['max_fre_length']}")
    
    # Prepare training data
    train_data, val_data = prepare_training_data(data_dict)
    
    # Create trainer with improved settings
    print("🏗️  Building model...")
    trainer = create_trainer(
        eng_vocab_size=data_dict['eng_vocab_size'],
        fre_vocab_size=data_dict['fre_vocab_size'],
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        learning_rate=learning_rate,
        device=device
    )
    
    total_params = sum(p.data.numel() for p in trainer.model.parameters())
    print(f"   🎯 Model created with {total_params:,} parameters")
    
    # Training with curriculum learning
    print("\\n🎓 Starting training with curriculum learning...")
    print("   📚 Phase 1 (0-30%): High teacher forcing (1.0 → 0.8)")
    print("   📚 Phase 2 (30-70%): Medium teacher forcing (0.8 → 0.5)")  
    print("   📚 Phase 3 (70-100%): Low teacher forcing (0.5 → 0.3)")
    print("-" * 60)
    
    history = trainer.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )
    
    print("\\n✅ Training completed!")
    
    # Plot training curves
    plot_training_history(history)
    
    return trainer, data_dict


def plot_training_history(history):
    """Plot training history with teacher forcing ratio"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Teacher forcing ratio
    ax3.plot(epochs, history['teacher_forcing_ratio'], 'g-', label='Teacher Forcing Ratio', linewidth=2)
    ax3.set_title('Teacher Forcing Ratio Schedule', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Ratio')
    ax3.set_ylim(0, 1.1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Combined view
    ax4_twin = ax4.twinx()
    ax4.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax4_twin.plot(epochs, history['teacher_forcing_ratio'], 'g--', label='Teacher Forcing Ratio', linewidth=2, alpha=0.7)
    ax4.set_title('Validation Loss vs Teacher Forcing Schedule', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Loss', color='r')
    ax4_twin.set_ylabel('Teacher Forcing Ratio', color='g')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_improved.png', dpi=300, bbox_inches='tight')
    print("📊 Training history plot saved as 'training_history_improved.png'")


def test_translations(trainer, data_dict, device='cpu'):
    """Test the trained model with various sentences"""
    
    print("\\n🔍 Testing Translations")
    print("=" * 60)
    
    # Test sentences
    test_sentences = [
        "hello",
        "how are you",
        "what is your name", 
        "i love you",
        "good morning",
        "thank you very much",
        "where is the bathroom",
        "i am hungry",
        "what time is it",
        "nice to meet you"
    ]
    
    eng_tokenizer = data_dict['eng_tokenizer']
    fre_tokenizer = data_dict['fre_tokenizer']
    max_eng_length = data_dict['max_eng_length']
    
    print("🎯 Greedy Decoding Results:")
    print("-" * 40)
    greedy_translations = evaluate_translations(
        trainer, test_sentences, eng_tokenizer, fre_tokenizer, 
        max_eng_length, device, use_beam_search=False
    )
    
    print("\\n🎯 Beam Search Results:")
    print("-" * 40)
    beam_translations = evaluate_translations(
        trainer, test_sentences, eng_tokenizer, fre_tokenizer, 
        max_eng_length, device, use_beam_search=True
    )
    
    print("\\n📊 Comparison Summary:")
    print("-" * 60)
    print(f"{'English':<20} {'Greedy':<25} {'Beam Search':<25}")
    print("-" * 70)
    for i, (eng, greedy, beam) in enumerate(zip(test_sentences, greedy_translations, beam_translations)):
        print(f"{eng:<20} {greedy:<25} {beam:<25}")
    
    return greedy_translations, beam_translations


def debug_single_translation(trainer, sentence, data_dict, device='cpu'):
    """Debug a single translation step by step"""
    
    print(f"\\n🔍 Debugging Translation: '{sentence}'")
    print("=" * 60)
    
    eng_tokenizer = data_dict['eng_tokenizer']
    fre_tokenizer = data_dict['fre_tokenizer'] 
    max_eng_length = data_dict['max_eng_length']
    
    # Detailed debugging
    translation = translate_sentence(
        trainer, sentence, eng_tokenizer, fre_tokenizer,
        max_eng_length, device, use_beam_search=False, debug=True
    )
    
    print(f"\\n🎯 Final Result: '{sentence}' → '{translation}'")
    
    return translation


if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'data_file_path': 'sample_english_french.csv',
        'epochs': 30,
        'batch_size': 32, 
        'embedding_dim': 256,
        'lstm_units': 256,
        'learning_rate': 0.001,
        'device': 'cpu',
        'sample_size': 1000  # Use subset for faster training
    }
    
    print("🔧 Training Configuration:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    
    # Train model
    trainer, data_dict = train_with_improvements(**CONFIG)
    
    # Test translations
    greedy_results, beam_results = test_translations(trainer, data_dict, CONFIG['device'])
    
    # Debug specific problematic cases
    problematic_sentences = ["hello", "what is your name"]
    for sentence in problematic_sentences:
        debug_single_translation(trainer, sentence, data_dict, CONFIG['device'])
    
    print("\\n🎉 Training and evaluation completed!")
    print("📋 Key Improvements Applied:")
    print("   ✅ Fixed decode_step method with proper state tracking")
    print("   ✅ Implemented scheduled sampling (teacher forcing curriculum)")
    print("   ✅ Added beam search decoding")
    print("   ✅ Improved EOS token handling") 
    print("   ✅ Added debugging capabilities")
    print("   ✅ Enhanced training monitoring")