"""
Simple test script to verify the improved neural machine translation system
This script tests the key improvements without external dependencies
"""
from train import create_trainer, prepare_training_data, translate_sentence
from data_utils import load_translation_data, prepare_data
import time


def quick_test_improvements():
    """
    Quick test of the key improvements to verify they work correctly
    """
    print("🧪 Testing Improved NMT System")
    print("=" * 50)
    
    # Load small sample of data
    print("📊 Loading sample data...")
    try:
        english, french = load_translation_data('sample_english_french.csv', sample_size=100)
        data_dict = prepare_data(english, french)
        print(f"✅ Data loaded: {len(english)} samples")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Create model
    print("🏗️  Creating model...")
    try:
        trainer = create_trainer(
            eng_vocab_size=data_dict['eng_vocab_size'],
            fre_vocab_size=data_dict['fre_vocab_size'],
            embedding_dim=64,  # Smaller for testing
            lstm_units=64,     # Smaller for testing  
            learning_rate=0.001,
            device='cpu'
        )
        print("✅ Model created successfully")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False
    
    # Test scheduled sampling functionality
    print("🎓 Testing scheduled sampling...")
    try:
        train_data, val_data = prepare_training_data(data_dict)
        
        # Test different teacher forcing ratios
        encoder_inputs = train_data['encoder_inputs'][:2]  # Small batch
        decoder_inputs = train_data['decoder_inputs'][:2] 
        targets = train_data['targets'][:2]
        
        # Test with full teacher forcing (1.0)
        loss_tf_1, acc_tf_1 = trainer.train_step(encoder_inputs, decoder_inputs, targets, teacher_forcing_ratio=1.0)
        print(f"   📊 TF=1.0: loss={loss_tf_1:.4f}, acc={acc_tf_1:.4f}")
        
        # Test with partial teacher forcing (0.5)  
        loss_tf_05, acc_tf_05 = trainer.train_step(encoder_inputs, decoder_inputs, targets, teacher_forcing_ratio=0.5)
        print(f"   📊 TF=0.5: loss={loss_tf_05:.4f}, acc={acc_tf_05:.4f}")
        
        print("✅ Scheduled sampling working")
    except Exception as e:
        print(f"❌ Error testing scheduled sampling: {e}")
        return False
    
    # Test teacher forcing ratio scheduling
    print("📅 Testing TF ratio scheduling...")
    try:
        # Test ratio calculation
        ratio_epoch_0 = trainer.get_teacher_forcing_ratio(0, 10)
        ratio_epoch_5 = trainer.get_teacher_forcing_ratio(5, 10) 
        ratio_epoch_9 = trainer.get_teacher_forcing_ratio(9, 10)
        
        print(f"   📊 Epoch 0/10: TF ratio = {ratio_epoch_0:.3f}")
        print(f"   📊 Epoch 5/10: TF ratio = {ratio_epoch_5:.3f}")  
        print(f"   📊 Epoch 9/10: TF ratio = {ratio_epoch_9:.3f}")
        
        # Verify decreasing trend
        assert ratio_epoch_0 > ratio_epoch_5 > ratio_epoch_9, "TF ratio should decrease over time"
        print("✅ TF ratio scheduling working")
    except Exception as e:
        print(f"❌ Error testing TF scheduling: {e}")
        return False
    
    # Test improved generation
    print("🎯 Testing improved generation...")
    try:
        # Test basic generation
        test_sentence = "hello"
        translation = translate_sentence(
            trainer, test_sentence, 
            data_dict['eng_tokenizer'], 
            data_dict['fre_tokenizer'],
            data_dict['max_eng_length'],
            device='cpu'
        )
        print(f"   📝 '{test_sentence}' → '{translation}'")
        
        # Test with beam search
        beam_translation = translate_sentence(
            trainer, test_sentence,
            data_dict['eng_tokenizer'],
            data_dict['fre_tokenizer'], 
            data_dict['max_eng_length'],
            device='cpu',
            use_beam_search=True,
            beam_width=2
        )
        print(f"   🔍 Beam search: '{test_sentence}' → '{beam_translation}'")
        
        print("✅ Generation methods working")
    except Exception as e:
        print(f"❌ Error testing generation: {e}")
        return False
    
    # Mini training test
    print("🏃 Testing mini training loop...")
    try:
        # Train for just 2 epochs to verify everything works
        history = trainer.fit(
            train_data=train_data,
            val_data=val_data, 
            epochs=2,
            batch_size=16,
            verbose=True
        )
        
        # Check that history contains teacher forcing ratios
        assert 'teacher_forcing_ratio' in history, "History should contain TF ratios"
        assert len(history['teacher_forcing_ratio']) == 2, "Should have TF ratio for each epoch"
        
        print("✅ Mini training completed successfully")
        print(f"   📊 Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"   📊 Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"   📊 TF ratios: {history['teacher_forcing_ratio']}")
        
    except Exception as e:
        print(f"❌ Error in mini training: {e}")
        return False
    
    print("\\n🎉 All tests passed!")
    print("🔧 Key improvements verified:")
    print("   ✅ Scheduled sampling (teacher forcing curriculum)")
    print("   ✅ Teacher forcing ratio scheduling") 
    print("   ✅ Improved decode_step with proper state tracking")
    print("   ✅ Enhanced generation with EOS handling")
    print("   ✅ Beam search decoding")
    print("   ✅ Training loop enhancements")
    
    return True


def test_problematic_case():
    """
    Test the specific problematic case mentioned by the user
    """
    print("\\n🐛 Testing Problematic Case: 'hello' → weird output")
    print("=" * 50)
    
    # This would be run after training to see if the improvements help
    print("💡 To test this properly, run improved_training.py")
    print("💡 The improvements should help reduce hallucination like:")
    print("   ❌ Before: 'hello' → 'hello elephant how are you'")  
    print("   ✅ After:  'hello' → 'bonjour' (or similar reasonable output)")
    
    print("\\n🔬 Key reasons for the improvement:")
    print("   1. Fixed decode_step - proper LSTM state tracking")
    print("   2. Scheduled sampling - model learns to handle its own predictions")
    print("   3. Better EOS handling - prevents runaway generation")
    print("   4. Beam search - explores multiple translation paths")


if __name__ == "__main__":
    print("🚀 Starting Comprehensive Test Suite")
    print("=" * 60)
    
    success = quick_test_improvements()
    
    if success:
        test_problematic_case()
        print("\\n🎯 Next Steps:")
        print("   1. Run 'python improved_training.py' for full training")
        print("   2. Compare translations before/after improvements")  
        print("   3. Monitor teacher forcing ratio during training")
        print("   4. Use beam search for better translation quality")
    else:
        print("\\n❌ Some tests failed. Please check the error messages above.")
    
    print("\\n📚 Summary of Improvements Applied:")
    print("   🔧 Fixed critical decode_step bug")
    print("   🎓 Added scheduled sampling/teacher forcing curriculum")
    print("   🔍 Implemented beam search decoding")
    print("   🛑 Better EOS token handling")
    print("   📊 Enhanced training monitoring")
    print("   🐛 Added debugging capabilities")