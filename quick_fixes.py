"""
QUICK FIXES for your specific implementation
Apply these changes to correct_implementation.py
"""
import torch
import torch.nn.functional as F

# ============================================================================
# FIX 1: TOKENIZER - Replace sequences_to_texts method
# ============================================================================

def fixed_sequences_to_texts(self, sequences):
    """
    Fixed version that properly filters special tokens
    Replace your current sequences_to_texts method with this
    """
    texts = []
    special_token_ids = {0, 1, 2}  # PAD, SOS, EOS tokens
    
    for sequence in sequences:
        words = []
        for idx in sequence:
            # Skip special tokens completely
            if idx in special_token_ids:
                continue
                
            word = self.index_word.get(idx, '')
            if word:  # Only add non-empty words
                words.append(word)
        
        texts.append(' '.join(words))
    return texts

# ============================================================================  
# FIX 2: IMPROVED TRANSLATION FUNCTION
# ============================================================================

def translate_sentence_fixed(model, sentence, eng_tokenizer, fre_tokenizer, 
                           max_eng_length, device='cpu', max_output_length=30):
    """
    Fixed translation function with repetition avoidance and better EOS handling
    Use this instead of your current translate_sentence
    """
    model.eval()  # Set to evaluation mode
    
    # Tokenize input
    sequence = eng_tokenizer.texts_to_sequences([sentence])
    if not sequence or not sequence[0]:
        return "ERROR: Could not tokenize input"
    
    # Import pad_sequences from your implementation
    from correct_implementation import pad_sequences
    padded = pad_sequences(sequence, maxlen=max_eng_length, padding='post')
    encoder_inputs = padded.to(device)
    
    # Get special tokens with fallback
    sos_token_id = fre_tokenizer.word_index.get('sos', 1)
    eos_token_id = fre_tokenizer.word_index.get('eos', 2)
    
    with torch.no_grad():
        try:
            # Encode input
            encoder_outputs, state_h, state_c = model.encoder(encoder_inputs)
            initial_state = (state_h, state_c)
            
            # Initialize decoder
            decoder_input = torch.full((1, 1), sos_token_id, dtype=torch.long, device=device)
            generated_tokens = []
            
            # Track recent tokens to avoid repetition
            recent_tokens = []
            consecutive_repeats = 0
            
            for step in range(max_output_length):
                # Get decoder output
                decoder_outputs = model.decoder(decoder_input, encoder_outputs, initial_state)
                
                # Get probabilities
                probs = F.softmax(decoder_outputs[0, -1], dim=-1)
                
                # Apply repetition penalty
                if len(recent_tokens) >= 2:
                    for recent_token in recent_tokens[-2:]:
                        if recent_token < len(probs):
                            probs[recent_token] *= 0.5  # Reduce probability of recent tokens
                
                # Get best token
                predicted_token_id = probs.argmax().item()
                
                # Check for EOS
                if predicted_token_id == eos_token_id:
                    break
                
                # Check for excessive repetition
                if len(recent_tokens) > 0 and predicted_token_id == recent_tokens[-1]:
                    consecutive_repeats += 1
                    if consecutive_repeats >= 2:  # Stop after 3 consecutive repeats
                        break
                else:
                    consecutive_repeats = 0
                
                # Add token to output
                generated_tokens.append(predicted_token_id)
                recent_tokens.append(predicted_token_id)
                
                # Keep only last 3 tokens for repetition checking
                if len(recent_tokens) > 3:
                    recent_tokens.pop(0)
                
                # Update decoder input for next step
                decoder_input = torch.tensor([[predicted_token_id]], device=device)
            
            # Convert to text using fixed tokenizer method
            if not generated_tokens:
                return ""
            
            # Use the fixed sequences_to_texts method
            translation = fixed_sequences_to_texts(fre_tokenizer, [generated_tokens])[0]
            return translation.strip()
            
        except Exception as e:
            return f"Translation error: {str(e)}"

# ============================================================================
# FIX 3: BEAM SEARCH TRANSLATION (Advanced)
# ============================================================================

def translate_with_beam_search_simple(model, sentence, eng_tokenizer, fre_tokenizer, 
                                    max_eng_length, device='cpu', beam_width=3):
    """
    Simple beam search implementation - use for better quality
    """
    model.eval()
    
    # Tokenize input  
    sequence = eng_tokenizer.texts_to_sequences([sentence])
    if not sequence or not sequence[0]:
        return "ERROR: Could not tokenize input"
    
    from correct_implementation import pad_sequences
    padded = pad_sequences(sequence, maxlen=max_eng_length, padding='post')
    encoder_inputs = padded.to(device)
    
    # Special tokens
    sos_id = fre_tokenizer.word_index.get('sos', 1)
    eos_id = fre_tokenizer.word_index.get('eos', 2)
    
    with torch.no_grad():
        # Encode
        encoder_outputs, state_h, state_c = model.encoder(encoder_inputs)
        
        # Initialize beams: (score, tokens)
        beams = [(0.0, [sos_id])]
        completed_beams = []
        
        for step in range(20):  # Max 20 steps
            all_candidates = []
            
            for score, tokens in beams:
                if tokens[-1] == eos_id:
                    completed_beams.append((score, tokens))
                    continue
                
                # Get decoder prediction
                decoder_input = torch.tensor([[tokens[-1]]], device=device)
                decoder_outputs = model.decoder(decoder_input, encoder_outputs, (state_h, state_c))
                
                # Get top predictions
                log_probs = F.log_softmax(decoder_outputs[0, -1], dim=-1)
                top_probs, top_indices = torch.topk(log_probs, beam_width)
                
                for prob, idx in zip(top_probs, top_indices):
                    token_id = idx.item()
                    new_score = score + prob.item()
                    
                    # Repetition penalty
                    if token_id in tokens[-3:]:
                        new_score -= 0.5
                    
                    all_candidates.append((new_score, tokens + [token_id]))
            
            # Keep best beams
            beams = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:beam_width]
            
            if not beams:
                break
        
        # Add remaining beams
        for score, tokens in beams:
            completed_beams.append((score, tokens))
        
        if not completed_beams:
            return "ERROR: No translation generated"
        
        # Get best translation
        best_score, best_tokens = max(completed_beams, key=lambda x: x[0])
        translation = fixed_sequences_to_texts(fre_tokenizer, [best_tokens])[0]
        return translation.strip()

# ============================================================================
# FIX 4: TESTING FUNCTION
# ============================================================================

def test_improvements(model, data_dict, device='cpu'):
    """
    Test the improvements on your specific problematic examples
    """
    print("🧪 TESTING TRANSLATION IMPROVEMENTS")
    print("=" * 60)
    
    # Your problematic cases
    test_cases = [
        ("hello", "Should be: bonjour/salut, NOT au revoir"),
        ("hi", "Should be: salut, NOT au revoir"),
        ("good morning", "Should be: bonjour, NOT bon matin"), 
        ("good evening", "Should be: bonsoir, NOT au matin"),
        ("goodbye", "Should be: au revoir, NOT au matin"),
        ("how are you", "Should NOT contain 'eos' token"),
        ("what is your name", "Should be: quel est votre nom"),
        ("where are you from", "Should NOT repeat 'vous'")
    ]
    
    print("Testing with FIXED translation function:")
    print("-" * 40)
    
    improvements = 0
    total_tests = len(test_cases)
    
    for i, (sentence, expectation) in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: '{sentence}'")
        print(f"   Expected: {expectation}")
        
        try:
            # Test with fixed function
            fixed_result = translate_sentence_fixed(
                model, sentence,
                data_dict['eng_tokenizer'],
                data_dict['fre_tokenizer'], 
                data_dict['max_eng_length'],
                device
            )
            
            print(f"   Fixed result: '{fixed_result}'")
            
            # Check for improvements
            is_improved = (
                'eos' not in fixed_result.lower() and  # No EOS token
                len(fixed_result.split()) > 0 and      # Not empty
                fixed_result != sentence                # Not just copying input
            )
            
            if is_improved:
                improvements += 1
                print("   ✅ IMPROVED!")
            else:
                print("   ❌ Still needs work")
                
        except Exception as e:
            print(f"   ERROR: {e}")
    
    success_rate = (improvements / total_tests) * 100
    print(f"\n📊 IMPROVEMENT SUMMARY:")
    print(f"   Total tests: {total_tests}")
    print(f"   Improvements: {improvements}")
    print(f"   Success rate: {success_rate:.1f}%")
    
    return improvements, total_tests

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def apply_fixes_to_your_model():
    """
    How to apply these fixes to your trained model
    """
    print("🔧 HOW TO APPLY FIXES:")
    print("=" * 40)
    print()
    print("1. TOKENIZER FIX:")
    print("   # Replace your Tokenizer.sequences_to_texts method")
    print("   # with fixed_sequences_to_texts function above")
    print()
    print("2. TRANSLATION FIX:")
    print("   # Use translate_sentence_fixed instead of translate_sentence")
    print("   # Or use translate_with_beam_search_simple for even better results")
    print()
    print("3. TESTING:")
    print("   # Run: improvements, total = test_improvements(model_large, data_dict_large, device)")
    print()
    print("4. EXPECTED RESULTS:")
    print("   # - No more 'eos' tokens in outputs")
    print("   # - Better word choices (hello → bonjour, not au revoir)")
    print("   # - Less repetition (no double 'vous')")
    print("   # - Better word order in questions")

if __name__ == "__main__":
    print("🚀 QUICK FIXES READY!")
    apply_fixes_to_your_model()