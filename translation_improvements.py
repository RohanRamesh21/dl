"""
Translation Quality Improvements
Fixes for common NMT issues in your model
"""
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import re

# ================================
# 1. IMPROVED TOKENIZER WITH BETTER EOS HANDLING
# ================================

class ImprovedTokenizer:
    """Enhanced tokenizer with better special token handling"""
    
    def __init__(self, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True):
        self.filters = filters
        self.lower = lower
        self.word_index = {}
        self.index_word = {}
        self.oov_token = '<UNK>'
        
        # Reserve special token IDs
        self.special_tokens = {
            '<PAD>': 0,
            '<SOS>': 1, 
            '<EOS>': 2,
            '<UNK>': 3
        }
        
        # Initialize with special tokens
        for token, idx in self.special_tokens.items():
            self.word_index[token.lower()] = idx
            self.index_word[idx] = token.lower()
    
    def _preprocess_text(self, text):
        """Clean and preprocess text"""
        if self.lower:
            text = text.lower()
        
        # Remove excessive punctuation but keep basic ones
        text = re.sub(r'[^\w\s\'\-\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = text.strip()
        return text
    
    def fit_on_texts(self, texts):
        """Build vocabulary from texts with frequency threshold"""
        word_counts = defaultdict(int)
        
        for text in texts:
            words = self._preprocess_text(text).split()
            for word in words:
                if word:  # Skip empty strings
                    word_counts[word] += 1
        
        # Sort by frequency and add to vocabulary (skip very rare words)
        min_frequency = 2  # Words must appear at least twice
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        next_index = len(self.special_tokens)
        for word, count in sorted_words:
            if count >= min_frequency and word not in self.word_index:
                self.word_index[word] = next_index
                self.index_word[next_index] = word
                next_index += 1
        
        print(f"Vocabulary size: {len(self.word_index)}")
        print(f"Filtered out {len([w for w, c in sorted_words if c < min_frequency])} rare words")
    
    def texts_to_sequences(self, texts):
        """Convert texts to sequences with UNK handling"""
        sequences = []
        unk_id = self.word_index['<unk>']
        
        for text in texts:
            words = self._preprocess_text(text).split()
            sequence = [self.word_index.get(word, unk_id) for word in words if word]
            sequences.append(sequence)
        return sequences
    
    def sequences_to_texts(self, sequences):
        """Convert sequences to texts with special token filtering"""
        texts = []
        special_token_ids = set(self.special_tokens.values())
        
        for sequence in sequences:
            words = []
            for idx in sequence:
                if idx in special_token_ids:
                    continue  # Skip all special tokens
                word = self.index_word.get(idx, '')
                if word:
                    words.append(word)
            texts.append(' '.join(words))
        return texts


# ================================
# 2. IMPROVED TRANSLATION FUNCTION WITH BEAM SEARCH
# ================================

def translate_with_beam_search(model, sentence, eng_tokenizer, fre_tokenizer, 
                             max_eng_length, device='cpu', beam_width=3, 
                             max_output_length=30, length_penalty=0.6):
    """
    Beam search translation with length penalty and repetition avoidance
    """
    model.eval()
    
    # Tokenize input
    sequence = eng_tokenizer.texts_to_sequences([sentence])
    if not sequence or not sequence[0]:
        return "ERROR: Could not tokenize input"
    
    from correct_implementation import pad_sequences
    padded = pad_sequences(sequence, maxlen=max_eng_length, padding='post')
    encoder_inputs = padded.to(device)
    
    # Get special tokens
    sos_id = fre_tokenizer.word_index.get('<sos>', fre_tokenizer.word_index.get('sos', 1))
    eos_id = fre_tokenizer.word_index.get('<eos>', fre_tokenizer.word_index.get('eos', 2))
    
    # Encode input
    with torch.no_grad():
        encoder_outputs, state_h, state_c = model.encoder(encoder_inputs)
        initial_state = (state_h, state_c)
        
        # Initialize beams: (score, tokens, state)
        beams = [(0.0, [sos_id], initial_state)]
        completed_beams = []
        
        for step in range(max_output_length):
            all_candidates = []
            
            for score, tokens, state in beams:
                if tokens[-1] == eos_id:
                    completed_beams.append((score, tokens))
                    continue
                
                # Current decoder input
                decoder_input = torch.tensor([[tokens[-1]]], device=device)
                decoder_outputs = model.decoder(decoder_input, encoder_outputs, state)
                
                # Get top k predictions
                log_probs = F.log_softmax(decoder_outputs[0, -1], dim=-1)
                top_k_probs, top_k_indices = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    token_id = top_k_indices[i].item()
                    token_prob = top_k_probs[i].item()
                    
                    # Repetition penalty
                    if len(tokens) >= 3 and token_id in tokens[-3:]:
                        token_prob -= 1.0  # Penalty for recent repetition
                    
                    new_tokens = tokens + [token_id]
                    new_score = score + token_prob
                    
                    # Length penalty (encourage longer sequences)
                    if token_id == eos_id:
                        length_norm = ((len(new_tokens) + 5) / 6.0) ** length_penalty
                        new_score = new_score / length_norm
                    
                    all_candidates.append((new_score, new_tokens, state))
            
            # Keep only top beams
            beams = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:beam_width]
            
            if not beams:  # All beams completed
                break
        
        # Add remaining beams to completed
        for score, tokens, _ in beams:
            completed_beams.append((score, tokens))
        
        if not completed_beams:
            return "ERROR: No translation generated"
        
        # Select best beam
        best_score, best_tokens = max(completed_beams, key=lambda x: x[0])
        
        # Convert to text
        translation = fre_tokenizer.sequences_to_texts([best_tokens])[0]
        return translation.strip()


# ================================
# 3. IMPROVED TRAINING WITH BETTER TEACHER FORCING
# ================================

def train_step_with_scheduled_sampling(model, encoder_inputs, decoder_inputs, targets, 
                                     optimizer, epoch, total_epochs, clip_grad_norm=1.0):
    """
    Training step with scheduled sampling (gradual teacher forcing reduction)
    """
    optimizer.zero_grad()
    
    # Scheduled sampling: reduce teacher forcing over time
    teacher_forcing_ratio = max(0.3, 1.0 - (epoch / total_epochs) * 0.7)
    
    batch_size, seq_len = decoder_inputs.size()
    vocab_size = model.decoder.vocab_size
    
    # Initialize decoder input and outputs
    decoder_input = decoder_inputs[:, :1]  # Start with SOS token
    decoder_outputs = []
    
    # Step-by-step decoding with scheduled sampling
    for t in range(seq_len):
        # Get decoder prediction
        decoder_output = model.decoder(decoder_input, *model.encoder(encoder_inputs)[:1], 
                                     model.encoder(encoder_inputs)[1:])
        decoder_outputs.append(decoder_output)
        
        # Decide whether to use teacher forcing
        use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
        
        if use_teacher_forcing and t + 1 < seq_len:
            # Use ground truth
            decoder_input = decoder_inputs[:, t+1:t+2]
        else:
            # Use model prediction
            predicted_tokens = decoder_output.argmax(dim=-1)
            decoder_input = predicted_tokens
    
    # Concatenate all outputs
    predictions = torch.cat(decoder_outputs, dim=1)
    
    # Compute loss
    from correct_implementation import sparse_categorical_crossentropy
    loss = sparse_categorical_crossentropy(predictions, targets)
    
    # Backward pass with gradient clipping
    loss.backward()
    if clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
    optimizer.step()
    
    # Compute accuracy
    pred_tokens = predictions.argmax(dim=-1)
    mask = targets != 0  # Non-padding mask
    correct = (pred_tokens == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float() if mask.sum() > 0 else 0.0
    
    return loss.item(), accuracy.item(), teacher_forcing_ratio


# ================================
# 4. TRANSLATION EVALUATION METRICS
# ================================

def compute_bleu_score(reference, candidate, n=4):
    """
    Simple BLEU score computation for translation evaluation
    """
    from collections import Counter
    import math
    
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    if len(cand_tokens) == 0:
        return 0.0
    
    # Compute precision for each n-gram order
    precisions = []
    for i in range(1, n+1):
        ref_ngrams = Counter(get_ngrams(ref_tokens, i))
        cand_ngrams = Counter(get_ngrams(cand_tokens, i))
        
        overlap = sum(min(ref_ngrams[ng], cand_ngrams[ng]) for ng in cand_ngrams)
        total = sum(cand_ngrams.values())
        
        if total == 0:
            precisions.append(0)
        else:
            precisions.append(overlap / total)
    
    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(cand_tokens)))
    
    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0
    
    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    return bleu


def evaluate_translations(test_pairs, model, data_dict, device='cpu'):
    """
    Comprehensive evaluation of translation quality
    """
    results = {
        'bleu_scores': [],
        'translations': [],
        'errors': {'repetition': 0, 'empty': 0, 'eos_visible': 0}
    }
    
    for eng_sentence, expected_fre in test_pairs:
        try:
            # Generate translation
            translation = translate_with_beam_search(
                model, eng_sentence, 
                data_dict['eng_tokenizer'], 
                data_dict['fre_tokenizer'],
                data_dict['max_eng_length'], 
                device
            )
            
            # Compute BLEU score
            bleu = compute_bleu_score(expected_fre, translation)
            results['bleu_scores'].append(bleu)
            
            # Check for common errors
            if not translation.strip():
                results['errors']['empty'] += 1
            if 'eos' in translation.lower():
                results['errors']['eos_visible'] += 1
            
            # Check repetition
            words = translation.lower().split()
            if len(words) > 1 and len(set(words)) < len(words) * 0.7:
                results['errors']['repetition'] += 1
            
            results['translations'].append({
                'english': eng_sentence,
                'expected': expected_fre,
                'generated': translation,
                'bleu': bleu
            })
            
        except Exception as e:
            results['translations'].append({
                'english': eng_sentence,
                'expected': expected_fre,
                'generated': f"ERROR: {e}",
                'bleu': 0.0
            })
    
    # Summary statistics
    results['avg_bleu'] = np.mean(results['bleu_scores']) if results['bleu_scores'] else 0
    results['total_tests'] = len(test_pairs)
    
    return results


# ================================
# 5. DIAGNOSTIC FUNCTIONS
# ================================

def diagnose_model_outputs(model, data_dict, device='cpu', num_samples=10):
    """
    Diagnose model behavior with various inputs
    """
    print("🔍 MODEL DIAGNOSIS")
    print("=" * 50)
    
    # Test basic greetings (your problem area)
    test_cases = [
        ("hello", "bonjour"),
        ("goodbye", "au revoir"),
        ("good morning", "bonjour"),
        ("good evening", "bonsoir"),
        ("thank you", "merci"),
        ("how are you", "comment allez-vous"),
        ("what is your name", "quel est votre nom"),
        ("where are you from", "d'où venez-vous")
    ]
    
    print("Testing problematic translations:")
    for eng, expected_fre in test_cases:
        try:
            # Test with improved beam search
            beam_result = translate_with_beam_search(
                model, eng, data_dict['eng_tokenizer'], 
                data_dict['fre_tokenizer'], data_dict['max_eng_length'], device
            )
            
            # Test with original method
            from correct_implementation import translate_sentence
            original_result = translate_sentence(
                model, eng, data_dict['eng_tokenizer'], 
                data_dict['fre_tokenizer'], data_dict['max_eng_length'], device
            )
            
            print(f"\n'{eng}':")
            print(f"  Expected:     {expected_fre}")
            print(f"  Beam Search:  {beam_result}")
            print(f"  Original:     {original_result}")
            
        except Exception as e:
            print(f"  ERROR: {e}")


# ================================
# 6. TESTING FRAMEWORK
# ================================

def run_improvement_tests():
    """
    Test the improvements on your specific error cases
    """
    # Your problematic outputs
    your_outputs = [
        ("hello", "au revoir", "bonjour"),
        ("hi", "au revoir", "salut"),
        ("good morning", "bon matin", "bonjour"),
        ("good evening", "au matin", "bonsoir"),
        ("good night", "revoir matin", "bonne nuit"),
        ("goodbye", "au matin", "au revoir"),
        ("see you later", "à bientôt", "à bientôt"),  # This one was correct!
        ("how are you", "d'où vous eos", "comment allez-vous"),
        ("what is your name", "quel votre est nom", "quel est votre nom"),
        ("where are you from", "d'où venez vous vous", "d'où venez-vous")
    ]
    
    print("🧪 TESTING IMPROVEMENTS")
    print("=" * 60)
    
    improvements = 0
    total = len(your_outputs)
    
    for eng, old_output, expected in your_outputs:
        print(f"\nTest: '{eng}'")
        print(f"  Old output:   {old_output}")
        print(f"  Expected:     {expected}")
        
        # Here you would test with your actual model
        # For now, showing the framework
        print("  [Framework ready - integrate with your trained model]")
    
    return improvements, total

if __name__ == "__main__":
    print("🚀 Translation Improvements Ready!")
    print("\nNext steps:")
    print("1. Replace your tokenizer with ImprovedTokenizer")
    print("2. Use translate_with_beam_search instead of greedy decoding")
    print("3. Retrain with train_step_with_scheduled_sampling")
    print("4. Evaluate with compute_bleu_score")
    print("5. Run diagnose_model_outputs on your trained model")