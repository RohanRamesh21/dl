"""
Neural Machine Translation implemented from scratch using PyTorch tensors
Following the exact specifications: use PyTorch data structures, implement everything else from scratch
"""
import torch
import torch.nn.functional as F
import math
import time
import pandas as pd
import numpy as np
import sys
import os
from collections import Counter
import re
from sklearn.model_selection import train_test_split

# Smart tqdm import for both terminal and notebook environments
try:
    # Check if we're in a Jupyter environment
    if 'ipykernel' in sys.modules or 'IPython' in sys.modules:
        from tqdm.notebook import tqdm
        NOTEBOOK_ENV = True
    else:
        from tqdm import tqdm
        NOTEBOOK_ENV = False
except ImportError:
    from tqdm import tqdm
    NOTEBOOK_ENV = False


class Linear:
    """Linear layer implemented from scratch using PyTorch tensors"""
    
    def __init__(self, in_features, out_features, bias=True, device='cpu'):
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and bias as PyTorch tensors with requires_grad=True
        self.weight = torch.randn(out_features, in_features, device=device, requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weight)
        
        if bias:
            self.bias = torch.zeros(out_features, device=device, requires_grad=True)
        else:
            self.bias = None
    
    def __call__(self, x):
        """Forward pass: y = xW^T + b"""
        output = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output
    
    def parameters(self):
        """Return parameters for optimizer"""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params


class Embedding:
    """Embedding layer implemented from scratch using PyTorch tensors"""
    
    def __init__(self, num_embeddings, embedding_dim, device='cpu'):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize embedding matrix as PyTorch tensor
        self.weight = torch.randn(num_embeddings, embedding_dim, device=device, requires_grad=True)
        torch.nn.init.xavier_normal_(self.weight)
    
    def __call__(self, input_ids):
        """Forward pass - index into embedding matrix"""
        # Use PyTorch's embedding function but with our custom weight matrix
        return F.embedding(input_ids, self.weight)
    
    def parameters(self):
        return [self.weight]


class LSTMCell:
    """LSTM cell implemented from scratch using PyTorch tensors"""
    
    def __init__(self, input_size, hidden_size, device='cpu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        
        # Input-to-hidden weights (implemented as Linear layers from scratch)
        self.W_ii = Linear(input_size, hidden_size, bias=True, device=device)  # input gate
        self.W_if = Linear(input_size, hidden_size, bias=True, device=device)  # forget gate
        self.W_ig = Linear(input_size, hidden_size, bias=True, device=device)  # cell gate
        self.W_io = Linear(input_size, hidden_size, bias=True, device=device)  # output gate
        
        # Hidden-to-hidden weights
        self.W_hi = Linear(hidden_size, hidden_size, bias=False, device=device)
        self.W_hf = Linear(hidden_size, hidden_size, bias=False, device=device)
        self.W_hg = Linear(hidden_size, hidden_size, bias=False, device=device)
        self.W_ho = Linear(hidden_size, hidden_size, bias=False, device=device)
    
    def __call__(self, x, hidden_state=None):
        """Forward pass through LSTM cell"""
        batch_size = x.size(0)
        
        if hidden_state is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=self.device, requires_grad=False)
            c_prev = torch.zeros(batch_size, self.hidden_size, device=self.device, requires_grad=False)
        else:
            h_prev, c_prev = hidden_state
        
        # LSTM gates - implementing the math from scratch
        i_t = torch.sigmoid(self.W_ii(x) + self.W_hi(h_prev))  # input gate
        f_t = torch.sigmoid(self.W_if(x) + self.W_hf(h_prev))  # forget gate
        g_t = torch.tanh(self.W_ig(x) + self.W_hg(h_prev))     # cell gate  
        o_t = torch.sigmoid(self.W_io(x) + self.W_ho(h_prev))  # output gate
        
        # Cell state and hidden state updates
        c_new = f_t * c_prev + i_t * g_t
        h_new = o_t * torch.tanh(c_new)
        
        return h_new, c_new
    
    def parameters(self):
        params = []
        for layer in [self.W_ii, self.W_if, self.W_ig, self.W_io,
                     self.W_hi, self.W_hf, self.W_hg, self.W_ho]:
            params.extend(layer.parameters())
        return params


class LSTM:
    """LSTM layer implemented from scratch using PyTorch tensors"""
    
    def __init__(self, input_size, hidden_size, batch_first=True, 
                 return_sequences=True, return_state=False, device='cpu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.device = device
        
        self.cell = LSTMCell(input_size, hidden_size, device)
    
    def __call__(self, x, initial_state=None):
        """Forward pass through LSTM"""
        if not self.batch_first:
            x = x.transpose(0, 1)  # Convert to batch_first
        
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Initialize states
        if initial_state is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=self.device)
            c_0 = torch.zeros(batch_size, self.hidden_size, device=self.device)
            initial_state = (h_0, c_0)
        
        outputs = []
        h_t, c_t = initial_state
        
        # Process sequence step by step
        for t in range(seq_len):
            h_t, c_t = self.cell(x[:, t, :], (h_t, c_t))
            if self.return_sequences:
                outputs.append(h_t.unsqueeze(1))
        
        if self.return_sequences:
            output = torch.cat(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
            if not self.batch_first:
                output = output.transpose(0, 1)
        else:
            output = h_t  # Only last timestep
        
        if self.return_state:
            return output, (h_t, c_t)
        else:
            return output
    
    def parameters(self):
        return self.cell.parameters()


class AdditiveAttention:
    """Additive (Bahdanau) attention implemented from scratch using PyTorch tensors"""
    
    def __init__(self, use_scale=True):
        self.use_scale = use_scale
    
    def __call__(self, query, value, key=None):
        """
        Compute additive attention
        query: (batch_size, query_len, hidden_size)
        value: (batch_size, value_len, hidden_size)
        key: (batch_size, value_len, hidden_size) - defaults to value
        """
        if key is None:
            key = value
        
        batch_size, query_len, hidden_size = query.shape
        value_len = value.size(1)
        
        # Additive attention: score = v^T * tanh(W_q * query + W_k * key)
        # Simplified version: just add query and key then sum
        
        # Expand dimensions for broadcasting
        query_expanded = query.unsqueeze(2)  # (batch, query_len, 1, hidden)
        key_expanded = key.unsqueeze(1)      # (batch, 1, value_len, hidden)
        
        # Compute additive scores
        scores = (query_expanded + key_expanded).sum(dim=-1)  # (batch, query_len, value_len)
        
        # Apply scaling
        if self.use_scale:
            scores = scores / math.sqrt(hidden_size)
        
        # Compute attention weights using softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context_vector = torch.matmul(attention_weights, value)
        
        return context_vector, attention_weights


class Encoder:
    """LSTM Encoder implemented from scratch"""
    
    def __init__(self, vocab_size, embedding_dim, lstm_units, device='cpu'):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.device = device
        
        # Components implemented from scratch
        self.embedding = Embedding(vocab_size, embedding_dim, device)
        self.lstm = LSTM(embedding_dim, lstm_units, batch_first=True, 
                        return_sequences=True, return_state=True, device=device)
    
    def __call__(self, x):
        """Forward pass through encoder"""
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM with state return
        encoder_outputs, (state_h, state_c) = self.lstm(embedded)
        
        return encoder_outputs, state_h, state_c
    
    def parameters(self):
        params = []
        params.extend(self.embedding.parameters())
        params.extend(self.lstm.parameters())
        return params


class Decoder:
    """LSTM Decoder with Attention implemented from scratch"""
    
    def __init__(self, vocab_size, embedding_dim, lstm_units, device='cpu'):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.device = device
        
        # Components implemented from scratch
        self.embedding = Embedding(vocab_size, embedding_dim, device)
        self.lstm = LSTM(embedding_dim, lstm_units, batch_first=True,
                        return_sequences=True, return_state=True, device=device)
        self.attention = AdditiveAttention(use_scale=True)
        self.output_dense = Linear(lstm_units * 2, vocab_size, device=device)  # *2 for concatenation
    
    def __call__(self, x, encoder_outputs, initial_state):
        """Forward pass through decoder"""
        # Embedding
        embedded = self.embedding(x)  # (batch_size, target_seq_len, embedding_dim)
        
        # LSTM forward pass with initial state
        lstm_outputs, _ = self.lstm(embedded, initial_state)
        # lstm_outputs: (batch_size, target_seq_len, lstm_units)
        
        # Apply attention mechanism
        context_vector, attention_weights = self.attention(
            query=lstm_outputs,      # (batch_size, target_seq_len, lstm_units)
            value=encoder_outputs,   # (batch_size, src_seq_len, lstm_units)
            key=encoder_outputs      # (batch_size, src_seq_len, lstm_units)
        )
        
        # Concatenate context vector and decoder outputs
        concatenated = torch.cat([context_vector, lstm_outputs], dim=-1)
        # concatenated: (batch_size, target_seq_len, lstm_units * 2)
        
        # Final dense layer (TimeDistributed equivalent)
        batch_size, seq_len, concat_dim = concatenated.size()
        concatenated_flat = concatenated.reshape(batch_size * seq_len, concat_dim)
        output_flat = self.output_dense(concatenated_flat)
        decoder_outputs = output_flat.reshape(batch_size, seq_len, self.vocab_size)
        
        return decoder_outputs
    
    def step_by_step_decode(self, encoder_outputs, initial_state, target_sequence=None, 
                           teacher_forcing_ratio=1.0, max_length=50, device='cpu'):
        """Step-by-step decoding with configurable teacher forcing ratio"""
        batch_size = encoder_outputs.size(0)
        sos_token_id = 1  # Assume SOS token ID is 1
        eos_token_id = 2  # Assume EOS token ID is 2
        
        # Initialize with SOS token
        decoder_input = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=device)
        decoder_state = initial_state
        outputs = []
        
        max_len = target_sequence.size(1) if target_sequence is not None else max_length
        
        for t in range(max_len):
            # Get decoder output for current step
            embedded = self.embedding(decoder_input)  # (batch_size, 1, embedding_dim)
            lstm_output, decoder_state = self.lstm(embedded, decoder_state)
            
            # Apply attention
            context_vector, attention_weights = self.attention(
                query=lstm_output,      # (batch_size, 1, lstm_units)
                value=encoder_outputs,  # (batch_size, src_seq_len, lstm_units)
                key=encoder_outputs     # (batch_size, src_seq_len, lstm_units)
            )
            
            # Concatenate and get predictions
            concatenated = torch.cat([context_vector, lstm_output], dim=-1)
            # Flatten for dense layer
            concatenated_flat = concatenated.reshape(batch_size, self.lstm_units * 2)
            step_output = self.output_dense(concatenated_flat)  # (batch_size, vocab_size)
            outputs.append(step_output.unsqueeze(1))  # Add time dimension
            
            # Decide next input (teacher forcing vs own prediction)
            if target_sequence is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Use ground truth (teacher forcing)
                if t + 1 < target_sequence.size(1):
                    decoder_input = target_sequence[:, t+1:t+2]  # Next ground truth token
                else:
                    break
            else:
                # Use own prediction
                predicted_token = step_output.argmax(dim=-1, keepdim=True)  # (batch_size, 1)
                decoder_input = predicted_token
                
                # Stop if all sequences hit EOS
                if torch.all(predicted_token.squeeze(-1) == eos_token_id):
                    break
        
        # Concatenate all outputs
        if outputs:
            return torch.cat(outputs, dim=1)  # (batch_size, seq_len, vocab_size)
        else:
            return torch.zeros((batch_size, 1, self.vocab_size), device=device)
    
    def parameters(self):
        params = []
        params.extend(self.embedding.parameters())
        params.extend(self.lstm.parameters())
        params.extend(self.output_dense.parameters())
        return params


class EncoderDecoderModel:
    """Complete Encoder-Decoder Model with Attention implemented from scratch"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=256, lstm_units=256, device='cpu'):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.device = device
        
        # Encoder and Decoder implemented from scratch
        self.encoder = Encoder(src_vocab_size, embedding_dim, lstm_units, device)
        self.decoder = Decoder(tgt_vocab_size, embedding_dim, lstm_units, device)
    
    def __call__(self, encoder_inputs, decoder_inputs):
        """Forward pass through the complete model"""
        # Encoder forward pass
        encoder_outputs, state_h, state_c = self.encoder(encoder_inputs)
        
        # Use encoder final states as initial decoder states
        initial_state = (state_h, state_c)
        
        # Decoder forward pass
        decoder_outputs = self.decoder(decoder_inputs, encoder_outputs, initial_state)
        
        return decoder_outputs
    
    def forward_with_teacher_forcing(self, encoder_inputs, target_sequence, teacher_forcing_ratio=1.0):
        """Forward pass with configurable teacher forcing ratio"""
        # Encoder forward pass
        encoder_outputs, state_h, state_c = self.encoder(encoder_inputs)
        
        # Use encoder final states as initial decoder states
        initial_state = (state_h, state_c)
        
        # Step-by-step decoder forward pass with teacher forcing
        decoder_outputs = self.decoder.step_by_step_decode(
            encoder_outputs=encoder_outputs,
            initial_state=initial_state,
            target_sequence=target_sequence,
            teacher_forcing_ratio=teacher_forcing_ratio,
            device=self.device
        )
        
        return decoder_outputs
    
    def parameters(self):
        """Return all parameters for optimizer"""
        params = []
        params.extend(self.encoder.parameters())
        params.extend(self.decoder.parameters())
        return params
    
    def to(self, device):
        """Move all parameters to device"""
        for param in self.parameters():
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad = param.grad.to(device)
        
        self.device = device
        self.encoder.device = device
        self.decoder.device = device
        self.encoder.embedding.device = device
        self.decoder.embedding.device = device
        self.encoder.lstm.device = device
        self.decoder.lstm.device = device
        return self


# Data preprocessing utilities (using PyTorch tensors)
class Tokenizer:
    """Custom tokenizer matching TensorFlow's behavior"""
    
    def __init__(self, filters='', lower=True):
        self.filters = filters
        self.lower = lower
        self.word_index = {}
        self.index_word = {}
        self.word_counts = Counter()
        self._index_counter = 1  # Reserve 0 for padding
    
    def _preprocess_text(self, text):
        if self.lower:
            text = text.lower()
        if self.filters:
            translator = str.maketrans('', '', self.filters)
            text = text.translate(translator)
        return text
    
    def fit_on_texts(self, texts):
        for text in texts:
            words = self._preprocess_text(text).split()
            for word in words:
                self.word_counts[word] += 1
        
        # Create word index based on frequency
        sorted_words = sorted(self.word_counts.items(), key=lambda x: (-x[1], x[0]))
        
        for word, count in sorted_words:
            if word not in self.word_index:
                self.word_index[word] = self._index_counter
                self.index_word[self._index_counter] = word
                self._index_counter += 1
    
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            words = self._preprocess_text(text).split()
            sequence = [self.word_index.get(word, 0) for word in words]  # 0 for unknown
            sequences.append(sequence)
        return sequences
    
    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            words = [self.index_word.get(idx, '') for idx in sequence if idx > 0]
            text = ' '.join(words)
            texts.append(text)
        return texts


def pad_sequences(sequences, maxlen=None, padding='post', value=0):
    """Pad sequences using PyTorch tensors"""
    if not sequences:
        return torch.empty((0, 0), dtype=torch.long)
    
    lengths = [len(seq) for seq in sequences]
    if maxlen is None:
        maxlen = max(lengths)
    
    num_samples = len(sequences)
    padded = torch.full((num_samples, maxlen), value, dtype=torch.long)
    
    for idx, seq in enumerate(sequences):
        if not seq:
            continue
        
        seq = seq[:maxlen]  # Truncate if too long
        
        if padding == 'post':
            padded[idx, :len(seq)] = torch.tensor(seq)
        else:  # 'pre'
            padded[idx, -len(seq):] = torch.tensor(seq)
    
    return padded


# Training utilities
def sparse_categorical_crossentropy(predictions, targets):
    """Sparse categorical crossentropy loss using PyTorch tensors"""
    # predictions: (batch_size, seq_len, vocab_size)
    # targets: (batch_size, seq_len)
    
    batch_size, seq_len, vocab_size = predictions.size()
    
    # Flatten for cross entropy calculation
    predictions_flat = predictions.reshape(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
    targets_flat = targets.reshape(-1)  # (batch_size * seq_len)
    
    # Use PyTorch's cross entropy (includes softmax)
    loss = F.cross_entropy(predictions_flat, targets_flat, ignore_index=0)  # ignore padding
    
    return loss


def create_training_data(fre_padded_sequences):
    """Create decoder input and target data"""
    # Decoder input: remove EOS token (last token)
    decoder_input_data = fre_padded_sequences[:, :-1]
    
    # Decoder target: remove SOS token (first token)
    decoder_target_data = fre_padded_sequences[:, 1:]
    
    return decoder_input_data, decoder_target_data


def train_step(model, encoder_inputs, decoder_inputs, targets, optimizer, teacher_forcing_ratio=1.0, clip_grad_norm=1.0):
    """Single training step with teacher forcing ratio and gradient clipping"""
    optimizer.zero_grad()
    
    # Forward pass with teacher forcing
    if teacher_forcing_ratio < 1.0:
        # Use step-by-step decoding with teacher forcing ratio
        target_sequence = torch.cat([decoder_inputs, targets[:, -1:]], dim=1)  # Add last target token
        predictions = model.forward_with_teacher_forcing(
            encoder_inputs, target_sequence, teacher_forcing_ratio
        )
        # Match target shape
        if predictions.size(1) > targets.size(1):
            predictions = predictions[:, :targets.size(1), :]
        elif predictions.size(1) < targets.size(1):
            # Pad predictions if needed
            pad_size = targets.size(1) - predictions.size(1)
            padding = torch.zeros(predictions.size(0), pad_size, predictions.size(2), 
                                device=predictions.device)
            predictions = torch.cat([predictions, padding], dim=1)
    else:
        # Standard forward pass (100% teacher forcing)
        predictions = model(encoder_inputs, decoder_inputs)
    
    # Compute loss
    loss = sparse_categorical_crossentropy(predictions, targets)
    
    # Compute accuracy
    pred_tokens = predictions.argmax(dim=-1)
    mask = targets != 0  # Non-padding mask
    correct = (pred_tokens == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float() if mask.sum() > 0 else 0.0
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    if clip_grad_norm > 0:
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        if total_norm > clip_grad_norm:
            clip_coef = clip_grad_norm / total_norm
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
    
    # Update parameters
    optimizer.step()
    
    return loss.item(), accuracy.item()


# Complete training pipeline
def prepare_data(data_file_path=None, sample_size=None, use_dummy_data=False):
    """Load and prepare translation data"""
    if use_dummy_data or data_file_path is None:
        # Use dummy data for testing
        english = np.array([
            "hello world", "how are you", "what is your name", "good morning",
            "thank you", "see you later", "have a nice day", "I am fine",
            "where are you from", "what time is it", "nice to meet you", "goodbye"
        ])
        french = np.array([
            "bonjour monde", "comment allez vous", "quel est votre nom", "bon matin",
            "merci", "à bientôt", "bonne journée", "je vais bien", 
            "d'où venez vous", "quelle heure est il", "enchanté de vous rencontrer", "au revoir"
        ])
    else:
        # Load real dataset
        print(f"Loading data from {data_file_path}...")
        try:
            dataset = pd.read_csv(data_file_path)
            print(f"Dataset shape: {dataset.shape}")
            print(f"Columns: {dataset.columns.tolist()}")
            
            # Handle different possible column names
            eng_col = None
            fre_col = None
            
            for col in dataset.columns:
                if 'english' in col.lower() or 'eng' in col.lower():
                    eng_col = col
                if 'french' in col.lower() or 'fre' in col.lower():
                    fre_col = col
            
            if eng_col is None or fre_col is None:
                print("Warning: Could not find English/French columns, using first two columns")
                eng_col = dataset.columns[0]
                fre_col = dataset.columns[1]
            
            print(f"Using columns: English='{eng_col}', French='{fre_col}'")
            
            # Clean data
            dataset = dataset.dropna(subset=[eng_col, fre_col])
            dataset = dataset.drop_duplicates(subset=[eng_col, fre_col])
            
            print(f"After cleaning: {len(dataset)} samples")
            
            if sample_size and sample_size < len(dataset):
                dataset = dataset.sample(sample_size, random_state=42)
                print(f"Sampled {sample_size} examples")
            
            english = np.array(dataset[eng_col])
            french = np.array(dataset[fre_col])
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Falling back to dummy data...")
            return prepare_data(use_dummy_data=True)
    
    # Add SOS and EOS tokens to French sentences
    french = np.array(['sos ' + sent + ' eos' for sent in french])
    
    print(f"Total samples: {len(english)}")
    print(f"Sample English: {english[0]}")
    print(f"Sample French: {french[0]}")
    
    # Split data
    eng_train, eng_val, fre_train, fre_val = train_test_split(
        english, french, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(eng_train)}")
    print(f"Validation samples: {len(eng_val)}")
    
    # Create tokenizers
    eng_tokenizer = Tokenizer()
    fre_tokenizer = Tokenizer()
    
    eng_tokenizer.fit_on_texts(eng_train)
    fre_tokenizer.fit_on_texts(fre_train)
    
    # Convert to sequences
    eng_train_seq = eng_tokenizer.texts_to_sequences(eng_train)
    eng_val_seq = eng_tokenizer.texts_to_sequences(eng_val)
    fre_train_seq = fre_tokenizer.texts_to_sequences(fre_train)
    fre_val_seq = fre_tokenizer.texts_to_sequences(fre_val)
    
    # Calculate max lengths
    max_eng_length = max(len(seq) for seq in eng_train_seq)
    max_fre_length = max(len(seq) for seq in fre_train_seq)
    
    print(f"Max English length: {max_eng_length}")
    print(f"Max French length: {max_fre_length}")
    
    # Pad sequences
    eng_train_pad = pad_sequences(eng_train_seq, maxlen=max_eng_length, padding='post')
    eng_val_pad = pad_sequences(eng_val_seq, maxlen=max_eng_length, padding='post')
    fre_train_pad = pad_sequences(fre_train_seq, maxlen=max_fre_length, padding='post')
    fre_val_pad = pad_sequences(fre_val_seq, maxlen=max_fre_length, padding='post')
    
    return {
        'eng_train_pad': eng_train_pad,
        'eng_val_pad': eng_val_pad,
        'fre_train_pad': fre_train_pad,
        'fre_val_pad': fre_val_pad,
        'eng_tokenizer': eng_tokenizer,
        'fre_tokenizer': fre_tokenizer,
        'eng_vocab_size': len(eng_tokenizer.word_index) + 1,
        'fre_vocab_size': len(fre_tokenizer.word_index) + 1,
        'max_eng_length': max_eng_length,
        'max_fre_length': max_fre_length
    }


def train_model_enhanced(data_file_path=None, epochs=10, batch_size=64, embedding_dim=256,
                        lstm_units=256, learning_rate=0.001, device='cpu', sample_size=None, 
                        use_dummy_data=False, teacher_forcing_schedule='linear'):
    """Enhanced training pipeline with teacher forcing scheduling"""
    print("=" * 60)
    print("ENHANCED NEURAL MACHINE TRANSLATION TRAINING")
    print("With Teacher Forcing Ratio Scheduling")
    print("=" * 60)
    
    print("Loading and preprocessing data...")
    data_dict = prepare_data(data_file_path, sample_size, use_dummy_data)
    
    print(f"English vocabulary size: {data_dict['eng_vocab_size']}")
    print(f"French vocabulary size: {data_dict['fre_vocab_size']}")
    
    # Create model
    model = EncoderDecoderModel(
        src_vocab_size=data_dict['eng_vocab_size'],
        tgt_vocab_size=data_dict['fre_vocab_size'],
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        device=device
    )
    model.to(device)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    # Prepare training data
    dec_train_input, dec_train_target = create_training_data(data_dict['fre_train_pad'])
    dec_val_input, dec_val_target = create_training_data(data_dict['fre_val_pad'])
    
    # Move data to device
    eng_train = data_dict['eng_train_pad'].to(device)
    eng_val = data_dict['eng_val_pad'].to(device)
    dec_train_input = dec_train_input.to(device)
    dec_train_target = dec_train_target.to(device)
    dec_val_input = dec_val_input.to(device)
    dec_val_target = dec_val_target.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    print(f"Training on {len(eng_train)} samples")
    print(f"Validation on {len(eng_val)} samples")
    print(f"Teacher forcing schedule: {teacher_forcing_schedule}")
    print("-" * 60)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': [],
        'teacher_forcing_ratio': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Teacher forcing scheduling functions
    def get_teacher_forcing_ratio(epoch, total_epochs, schedule='linear'):
        progress = epoch / total_epochs
        if schedule == 'linear':
            return max(0.3, 1.0 - 0.7 * progress)  # 1.0 -> 0.3
        elif schedule == 'exponential':
            return max(0.3, 1.0 * (0.3 ** progress))  # Exponential decay
        elif schedule == 'step':
            if progress < 0.3:
                return 1.0
            elif progress < 0.7:
                return 0.7
            else:
                return 0.3
        else:  # constant
            return 1.0
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        # Get teacher forcing ratio for this epoch
        tf_ratio = get_teacher_forcing_ratio(epoch, epochs, teacher_forcing_schedule)
        print(f"Epoch {epoch+1}/{epochs} - Teacher forcing ratio: {tf_ratio:.3f}")
        
        # Training phase
        model.train = True
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        # Progress bar for training
        batch_indices = list(range(0, len(eng_train), batch_size))
        if NOTEBOOK_ENV:
            pbar = tqdm(total=len(batch_indices), desc=f'Training', leave=True, position=0)
        else:
            pbar = tqdm(total=len(batch_indices), desc=f'Training')
        
        for batch_idx, i in enumerate(batch_indices):
            end_i = min(i + batch_size, len(eng_train))
            
            enc_batch = eng_train[i:end_i]
            dec_input_batch = dec_train_input[i:end_i]
            dec_target_batch = dec_train_target[i:end_i]
            
            loss, acc = train_step(
                model, enc_batch, dec_input_batch, dec_target_batch, 
                optimizer, teacher_forcing_ratio=tf_ratio
            )
            
            total_loss += loss
            total_acc += acc
            num_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            avg_acc = total_acc / num_batches
            
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'acc': f'{acc:.4f}',
                'tf_ratio': f'{tf_ratio:.3f}'
            })
            pbar.update(1)
        
        pbar.close()
        
        avg_train_loss = total_loss / num_batches
        avg_train_acc = total_acc / num_batches
        
        # Validation phase (always use teacher forcing for consistency)
        model.train = False
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        
        val_batch_indices = list(range(0, len(eng_val), batch_size))
        
        with torch.no_grad():
            if NOTEBOOK_ENV:
                val_pbar = tqdm(total=len(val_batch_indices), desc='Validation', 
                               leave=True, position=0)
            else:
                val_pbar = tqdm(total=len(val_batch_indices), desc='Validation')
            
            for batch_idx, i in enumerate(val_batch_indices):
                end_i = min(i + batch_size, len(eng_val))
                
                enc_batch = eng_val[i:end_i]
                dec_input_batch = dec_val_input[i:end_i]
                dec_target_batch = dec_val_target[i:end_i]
                
                # Forward pass only (with 100% teacher forcing for stable validation)
                predictions = model(enc_batch, dec_input_batch)
                loss = sparse_categorical_crossentropy(predictions, dec_target_batch)
                
                # Compute accuracy
                pred_tokens = predictions.argmax(dim=-1)
                mask = dec_target_batch != 0
                correct = (pred_tokens == dec_target_batch) & mask
                accuracy = correct.sum().float() / mask.sum().float() if mask.sum() > 0 else 0.0
                
                val_loss += loss.item()
                val_acc += accuracy.item()
                val_batches += 1
                
                # Update validation progress bar
                avg_val_loss = val_loss / val_batches
                avg_val_acc = val_acc / val_batches
                val_pbar.set_postfix({
                    'val_loss': f'{avg_val_loss:.4f}',
                    'val_acc': f'{avg_val_acc:.4f}'
                })
                val_pbar.update(1)
            
            val_pbar.close()
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        avg_val_acc = val_acc / val_batches if val_batches > 0 else 0.0
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        history['learning_rate'].append(current_lr)
        history['teacher_forcing_ratio'].append(tf_ratio)
        
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f"Epoch {epoch+1:2d}/{epochs} - {epoch_time:.2f}s - "
              f"loss: {avg_train_loss:.4f} - acc: {avg_train_acc:.4f} - "
              f"val_loss: {avg_val_loss:.4f} - val_acc: {avg_val_acc:.4f} - "
              f"lr: {current_lr:.2e} - tf: {tf_ratio:.3f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 7:  # Increased patience for teacher forcing
                print(f"\nEarly stopping after {epoch+1} epochs (no improvement for 7 epochs)")
                break
    
    return model, data_dict, history
    """Load and prepare translation data"""
    if use_dummy_data or data_file_path is None:
        # Use dummy data for testing
        english = np.array([
            "hello world", "how are you", "what is your name", "good morning",
            "thank you", "see you later", "have a nice day", "I am fine",
            "where are you from", "what time is it", "nice to meet you", "goodbye"
        ])
        french = np.array([
            "bonjour monde", "comment allez vous", "quel est votre nom", "bon matin",
            "merci", "à bientôt", "bonne journée", "je vais bien", 
            "d'où venez vous", "quelle heure est il", "enchanté de vous rencontrer", "au revoir"
        ])
    else:
        # Load real dataset
        print(f"Loading data from {data_file_path}...")
        try:
            dataset = pd.read_csv(data_file_path)
            print(f"Dataset shape: {dataset.shape}")
            print(f"Columns: {dataset.columns.tolist()}")
            
            # Handle different possible column names
            eng_col = None
            fre_col = None
            
            for col in dataset.columns:
                if 'english' in col.lower() or 'eng' in col.lower():
                    eng_col = col
                if 'french' in col.lower() or 'fre' in col.lower():
                    fre_col = col
            
            if eng_col is None or fre_col is None:
                print("Warning: Could not find English/French columns, using first two columns")
                eng_col = dataset.columns[0]
                fre_col = dataset.columns[1]
            
            print(f"Using columns: English='{eng_col}', French='{fre_col}'")
            
            # Clean data
            dataset = dataset.dropna(subset=[eng_col, fre_col])
            dataset = dataset.drop_duplicates(subset=[eng_col, fre_col])
            
            print(f"After cleaning: {len(dataset)} samples")
            
            if sample_size and sample_size < len(dataset):
                dataset = dataset.sample(sample_size, random_state=42)
                print(f"Sampled {sample_size} examples")
            
            english = np.array(dataset[eng_col])
            french = np.array(dataset[fre_col])
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Falling back to dummy data...")
            return prepare_data(use_dummy_data=True)
    
    # Add SOS and EOS tokens to French sentences
    french = np.array(['sos ' + sent + ' eos' for sent in french])
    
    print(f"Total samples: {len(english)}")
    print(f"Sample English: {english[0]}")
    print(f"Sample French: {french[0]}")
    
    # Split data
    eng_train, eng_val, fre_train, fre_val = train_test_split(
        english, french, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(eng_train)}")
    print(f"Validation samples: {len(eng_val)}")
    
    # Create tokenizers
    eng_tokenizer = Tokenizer()
    fre_tokenizer = Tokenizer()
    
    eng_tokenizer.fit_on_texts(eng_train)
    fre_tokenizer.fit_on_texts(fre_train)
    
    # Convert to sequences
    eng_train_seq = eng_tokenizer.texts_to_sequences(eng_train)
    eng_val_seq = eng_tokenizer.texts_to_sequences(eng_val)
    fre_train_seq = fre_tokenizer.texts_to_sequences(fre_train)
    fre_val_seq = fre_tokenizer.texts_to_sequences(fre_val)
    
    # Calculate max lengths
    max_eng_length = max(len(seq) for seq in eng_train_seq)
    max_fre_length = max(len(seq) for seq in fre_train_seq)
    
    print(f"Max English length: {max_eng_length}")
    print(f"Max French length: {max_fre_length}")
    
    # Pad sequences
    eng_train_pad = pad_sequences(eng_train_seq, maxlen=max_eng_length, padding='post')
    eng_val_pad = pad_sequences(eng_val_seq, maxlen=max_eng_length, padding='post')
    fre_train_pad = pad_sequences(fre_train_seq, maxlen=max_fre_length, padding='post')
    fre_val_pad = pad_sequences(fre_val_seq, maxlen=max_fre_length, padding='post')
    
    return {
        'eng_train_pad': eng_train_pad,
        'eng_val_pad': eng_val_pad,
        'fre_train_pad': fre_train_pad,
        'fre_val_pad': fre_val_pad,
        'eng_tokenizer': eng_tokenizer,
        'fre_tokenizer': fre_tokenizer,
        'eng_vocab_size': len(eng_tokenizer.word_index) + 1,
        'fre_vocab_size': len(fre_tokenizer.word_index) + 1,
        'max_eng_length': max_eng_length,
        'max_fre_length': max_fre_length
    }


def train_model(data_file_path=None, epochs=10, batch_size=64, embedding_dim=256,
               lstm_units=256, learning_rate=0.001, device='cpu', sample_size=None, use_dummy_data=False):
    """Complete training pipeline with improvements"""
    print("=" * 60)
    print("NEURAL MACHINE TRANSLATION TRAINING")
    print("=" * 60)
    
    print("Loading and preprocessing data...")
    data_dict = prepare_data(data_file_path, sample_size, use_dummy_data)
    
    print(f"English vocabulary size: {data_dict['eng_vocab_size']}")
    print(f"French vocabulary size: {data_dict['fre_vocab_size']}")
    
    # Create model
    model = EncoderDecoderModel(
        src_vocab_size=data_dict['eng_vocab_size'],
        tgt_vocab_size=data_dict['fre_vocab_size'],
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        device=device
    )
    model.to(device)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    # Prepare training data
    dec_train_input, dec_train_target = create_training_data(data_dict['fre_train_pad'])
    dec_val_input, dec_val_target = create_training_data(data_dict['fre_val_pad'])
    
    # Move data to device
    eng_train = data_dict['eng_train_pad'].to(device)
    eng_val = data_dict['eng_val_pad'].to(device)
    dec_train_input = dec_train_input.to(device)
    dec_train_target = dec_train_target.to(device)
    dec_val_input = dec_val_input.to(device)
    dec_val_target = dec_val_target.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    print(f"Training on {len(eng_train)} samples")
    print(f"Validation on {len(eng_val)} samples")
    print("-" * 60)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train = True
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        # Single progress bar per epoch that updates with each batch
        batch_indices = list(range(0, len(eng_train), batch_size))
        if NOTEBOOK_ENV:
            pbar = tqdm(total=len(batch_indices), desc=f'Epoch {epoch+1}/{epochs}', 
                       leave=True, position=0)
        else:
            pbar = tqdm(total=len(batch_indices), desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, i in enumerate(batch_indices):
            end_i = min(i + batch_size, len(eng_train))
            
            enc_batch = eng_train[i:end_i]
            dec_input_batch = dec_train_input[i:end_i]
            dec_target_batch = dec_train_target[i:end_i]
            
            loss, acc = train_step(model, enc_batch, dec_input_batch, dec_target_batch, optimizer)
            
            total_loss += loss
            total_acc += acc
            num_batches += 1
            
            # Update the single progress bar with current metrics
            avg_loss = total_loss / num_batches
            avg_acc = total_acc / num_batches
            
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'acc': f'{acc:.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'avg_acc': f'{avg_acc:.4f}'
            })
            pbar.update(1)  # Move progress bar forward by 1 step
        
        pbar.close()  # Close the progress bar when epoch is done
        
        avg_train_loss = total_loss / num_batches
        avg_train_acc = total_acc / num_batches
        
        # Validation phase with single progress bar
        model.train = False
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        
        val_batch_indices = list(range(0, len(eng_val), batch_size))
        
        with torch.no_grad():
            if NOTEBOOK_ENV:
                val_pbar = tqdm(total=len(val_batch_indices), desc='Validation', 
                               leave=True, position=0)
            else:
                val_pbar = tqdm(total=len(val_batch_indices), desc='Validation')
            
            for batch_idx, i in enumerate(val_batch_indices):
                end_i = min(i + batch_size, len(eng_val))
                
                enc_batch = eng_val[i:end_i]
                dec_input_batch = dec_val_input[i:end_i]
                dec_target_batch = dec_val_target[i:end_i]
                
                # Forward pass only
                predictions = model(enc_batch, dec_input_batch)
                loss = sparse_categorical_crossentropy(predictions, dec_target_batch)
                
                # Compute accuracy
                pred_tokens = predictions.argmax(dim=-1)
                mask = dec_target_batch != 0
                correct = (pred_tokens == dec_target_batch) & mask
                accuracy = correct.sum().float() / mask.sum().float()
                
                val_loss += loss.item()
                val_acc += accuracy.item()
                val_batches += 1
                
                # Update validation progress bar
                avg_val_loss = val_loss / val_batches
                avg_val_acc = val_acc / val_batches
                val_pbar.set_postfix({
                    'val_loss': f'{avg_val_loss:.4f}',
                    'val_acc': f'{avg_val_acc:.4f}'
                })
                val_pbar.update(1)
            
            val_pbar.close()
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        avg_val_acc = val_acc / val_batches if val_batches > 0 else 0.0
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        history['learning_rate'].append(current_lr)
        
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f"Epoch {epoch+1:2d}/{epochs} - {epoch_time:.2f}s - "
              f"loss: {avg_train_loss:.4f} - acc: {avg_train_acc:.4f} - "
              f"val_loss: {avg_val_loss:.4f} - val_acc: {avg_val_acc:.4f} - lr: {current_lr:.2e}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:  # Stop if no improvement for 5 epochs
                print(f"\nEarly stopping after {epoch+1} epochs (no improvement for 5 epochs)")
                break
    
    return model, data_dict, history


def translate_sentence_beam_search(model, sentence, eng_tokenizer, fre_tokenizer, max_eng_length, 
                                  device='cpu', max_output_length=50, beam_width=3, repetition_penalty=2.0):
    """Translate using beam search with repetition penalty"""
    model.train = False
    
    # Tokenize and pad input
    sequence = eng_tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=max_eng_length, padding='post')
    encoder_inputs = padded.to(device)
    
    # Get special tokens
    sos_token_id = fre_tokenizer.word_index.get('sos', 1)
    eos_token_id = fre_tokenizer.word_index.get('eos', 2)
    
    # Encode input
    encoder_outputs, state_h, state_c = model.encoder(encoder_inputs)
    
    # Beam search
    beams = [(0.0, [sos_token_id], (state_h, state_c))]  # (score, tokens, state)
    completed_beams = []
    
    with torch.no_grad():
        for step in range(max_output_length):
            candidates = []
            
            for score, tokens, (h_state, c_state) in beams:
                if tokens[-1] == eos_token_id:
                    completed_beams.append((score, tokens))
                    continue
                
                # Get next token probabilities
                decoder_input = torch.tensor([[tokens[-1]]], device=device)
                decoder_outputs = model.decoder(decoder_input, encoder_outputs, (h_state, c_state))
                
                # Apply repetition penalty
                vocab_size = decoder_outputs.size(-1)
                token_probs = F.softmax(decoder_outputs.squeeze(0).squeeze(0), dim=-1)
                
                # Penalize repeated tokens
                for prev_token in set(tokens[1:]):  # Skip SOS token
                    if prev_token < vocab_size:
                        # Count occurrences of this token
                        count = tokens[1:].count(prev_token)
                        penalty = repetition_penalty ** count
                        token_probs[prev_token] = token_probs[prev_token] / penalty
                
                # Get top-k candidates
                top_probs, top_indices = torch.topk(token_probs, beam_width)
                
                for prob, token_id in zip(top_probs, top_indices):
                    new_score = score - torch.log(prob).item()  # Negative log likelihood
                    new_tokens = tokens + [token_id.item()]
                    
                    # Update LSTM state for next step
                    h_new, c_new = model.decoder.lstm.cell(
                        model.decoder.embedding(torch.tensor([[token_id]], device=device)).squeeze(0),
                        (h_state, c_state)
                    )
                    
                    candidates.append((new_score, new_tokens, (h_new, c_new)))
            
            # Keep only top beams
            beams = sorted(candidates, key=lambda x: x[0])[:beam_width]
            
            if not beams:  # All beams completed
                break
        
        # Add remaining beams to completed
        for score, tokens, _ in beams:
            completed_beams.append((score, tokens))
    
    if not completed_beams:
        return ""
    
    # Select best beam (lowest score = highest probability)
    best_tokens = min(completed_beams, key=lambda x: x[0])[1]
    
    # Remove SOS and EOS tokens
    if best_tokens[0] == sos_token_id:
        best_tokens = best_tokens[1:]
    if best_tokens and best_tokens[-1] == eos_token_id:
        best_tokens = best_tokens[:-1]
    
    # Convert to text
    translation = fre_tokenizer.sequences_to_texts([best_tokens])[0] if best_tokens else ""
    return translation.strip()


def translate_sentence_improved(model, sentence, eng_tokenizer, fre_tokenizer, max_eng_length, 
                               device='cpu', max_output_length=30, temperature=0.8, repetition_penalty=1.5):
    """Improved translation with temperature sampling and repetition penalty"""
    model.train = False
    
    # Tokenize and pad input
    sequence = eng_tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=max_eng_length, padding='post')
    encoder_inputs = padded.to(device)
    
    # Get special tokens
    sos_token_id = fre_tokenizer.word_index.get('sos', 1)
    eos_token_id = fre_tokenizer.word_index.get('eos', 2)
    
    # Encode input
    encoder_outputs, state_h, state_c = model.encoder(encoder_inputs)
    
    # Initialize generation
    decoder_input = torch.full((1, 1), sos_token_id, dtype=torch.long, device=device)
    generated_tokens = []
    
    with torch.no_grad():
        for step in range(max_output_length):
            # Get predictions
            decoder_outputs = model.decoder(decoder_input, encoder_outputs, (state_h, state_c))
            logits = decoder_outputs.squeeze(0).squeeze(0)  # (vocab_size,)
            
            # Apply repetition penalty
            for token in set(generated_tokens):
                if token < logits.size(0):
                    count = generated_tokens.count(token)
                    penalty = repetition_penalty ** count
                    logits[token] = logits[token] / penalty
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            
            # Use top-p sampling for better diversity
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Keep only top 90% probability mass
            top_p = 0.9
            cutoff_index = (cumulative_probs > top_p).nonzero(as_tuple=False)[0][0].item() + 1
            top_indices = sorted_indices[:cutoff_index]
            top_probs = sorted_probs[:cutoff_index]
            top_probs = top_probs / top_probs.sum()  # Renormalize
            
            # Sample from top-p distribution
            predicted_token_id = top_indices[torch.multinomial(top_probs, 1)].item()
            
            # Stop if EOS token
            if predicted_token_id == eos_token_id:
                break
            
            generated_tokens.append(predicted_token_id)
            
            # Early stopping for very repetitive sequences
            if len(generated_tokens) >= 4:
                last_4 = generated_tokens[-4:]
                if len(set(last_4)) == 1:  # All same token
                    break
            
            # Use predicted token as next input
            decoder_input = torch.tensor([[predicted_token_id]], device=device)
    
    # Convert tokens to text
    if not generated_tokens:
        return ""
    
    translation = fre_tokenizer.sequences_to_texts([generated_tokens])[0]
    return translation.strip()


def translate_sentence_robust(model, sentence, eng_tokenizer, fre_tokenizer, max_eng_length, 
                             device='cpu', max_output_length=30, temperature=0.7):
    """Robust translation that matches training step-by-step approach"""
    model.train = False
    
    # Tokenize and pad input
    sequence = eng_tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=max_eng_length, padding='post')
    encoder_inputs = padded.to(device)
    
    # Get special tokens
    sos_token_id = fre_tokenizer.word_index.get('sos', 1)
    eos_token_id = fre_tokenizer.word_index.get('eos', 2)
    
    # Encode input
    encoder_outputs, state_h, state_c = model.encoder(encoder_inputs)
    decoder_state = (state_h, state_c)
    
    # Initialize with SOS token
    decoder_input = torch.full((1, 1), sos_token_id, dtype=torch.long, device=device)
    generated_tokens = []
    
    with torch.no_grad():
        for step in range(max_output_length):
            # Get decoder output for current step (matching training)
            embedded = model.decoder.embedding(decoder_input)  # (1, 1, embedding_dim)
            lstm_output, decoder_state = model.decoder.lstm(embedded, decoder_state)
            
            # Apply attention (matching training)
            context_vector, _ = model.decoder.attention(
                query=lstm_output,      # (1, 1, lstm_units)
                value=encoder_outputs,  # (1, src_seq_len, lstm_units)
                key=encoder_outputs     # (1, src_seq_len, lstm_units)
            )
            
            # Concatenate and get predictions (matching training)
            concatenated = torch.cat([context_vector, lstm_output], dim=-1)
            concatenated_flat = concatenated.reshape(1, model.decoder.lstm_units * 2)
            step_output = model.decoder.output_dense(concatenated_flat)  # (1, vocab_size)
            
            # Apply temperature for better diversity
            if temperature != 1.0:
                step_output = step_output / temperature
            
            # Sample from distribution (with slight randomness)
            probs = F.softmax(step_output, dim=-1)
            
            # Use top-k sampling to prevent degenerate outputs
            top_k = 5
            top_probs, top_indices = torch.topk(probs, top_k)
            top_probs = top_probs / top_probs.sum()  # Renormalize
            
            # Sample from top-k
            predicted_token_id = top_indices[0, torch.multinomial(top_probs, 1)].item()
            
            # Stop if EOS token
            if predicted_token_id == eos_token_id:
                break
            
            # Avoid immediate repetition
            if generated_tokens and predicted_token_id == generated_tokens[-1]:
                # Pick second choice if available
                if len(top_indices[0]) > 1:
                    predicted_token_id = top_indices[0, 1].item()
            
            generated_tokens.append(predicted_token_id)
            
            # Use predicted token as next input
            decoder_input = torch.tensor([[predicted_token_id]], device=device)
    
    # Convert tokens to text
    if not generated_tokens:
        return ""
    
    translation = fre_tokenizer.sequences_to_texts([generated_tokens])[0]
    return translation.strip()


def translate_sentence(model, sentence, eng_tokenizer, fre_tokenizer, max_eng_length, device='cpu', max_output_length=50):
    """Enhanced translation function - uses robust method by default"""
    try:
        # Use robust translation that matches training
        return translate_sentence_robust(
            model, sentence, eng_tokenizer, fre_tokenizer, max_eng_length, device
        )
    except Exception as e:
        # Fall back to simple translation
        return translate_sentence_simple(model, sentence, eng_tokenizer, fre_tokenizer, max_eng_length, device)


def translate_sentence_simple(model, sentence, eng_tokenizer, fre_tokenizer, max_eng_length, device='cpu', max_output_length=20):
    """Simple greedy translation with early stopping"""
    model.train = False
    
    sequence = eng_tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=max_eng_length, padding='post')
    encoder_inputs = padded.to(device)
    
    sos_token_id = fre_tokenizer.word_index.get('sos', 1)
    eos_token_id = fre_tokenizer.word_index.get('eos', 2)
    
    encoder_outputs, state_h, state_c = model.encoder(encoder_inputs)
    
    decoder_input = torch.full((1, 1), sos_token_id, dtype=torch.long, device=device)
    generated_tokens = []
    prev_token = None
    repeat_count = 0
    
    with torch.no_grad():
        for _ in range(max_output_length):
            decoder_outputs = model.decoder(decoder_input, encoder_outputs, (state_h, state_c))
            predicted_token_id = decoder_outputs.argmax(dim=-1).item()
            
            if predicted_token_id == eos_token_id:
                break
            
            # Simple repetition check
            if predicted_token_id == prev_token:
                repeat_count += 1
                if repeat_count >= 2:  # Stop after 3 consecutive repeats
                    break
            else:
                repeat_count = 0
            
            generated_tokens.append(predicted_token_id)
            prev_token = predicted_token_id
            decoder_input = torch.tensor([[predicted_token_id]], device=device)
    
    if not generated_tokens:
        return ""
    
    translation = fre_tokenizer.sequences_to_texts([generated_tokens])[0]
    return translation.strip()


# Demo function
def demo():
    """Demonstrate the enhanced implementation with teacher forcing"""
    print("=" * 60)
    print("ENHANCED NEURAL MACHINE TRANSLATION FROM SCRATCH")
    print("Using PyTorch tensors with Teacher Forcing Ratio Training")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test with dummy data using enhanced training
    print("\n1. Testing with enhanced training (teacher forcing ratio scheduling)...")
    model, data_dict, history = train_model_enhanced(
        data_file_path=None,  # Use dummy data
        epochs=8,
        batch_size=4,  # Small batch for demo
        embedding_dim=64,  # Small for demo
        lstm_units=32,     # Small for demo
        learning_rate=0.01,
        device=device,
        use_dummy_data=True,
        teacher_forcing_schedule='linear'  # 1.0 -> 0.3 linearly
    )
    
    # Show training progress
    print("\n📊 Training Summary:")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final training accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final teacher forcing ratio: {history['teacher_forcing_ratio'][-1]:.3f}")
    
    # Test translation
    print("\n2. Testing enhanced translation:")
    test_sentences = ["hello world", "how are you", "thank you", "good morning"]
    
    for sentence in test_sentences:
        translation = translate_sentence(
            model=model,
            sentence=sentence,
            eng_tokenizer=data_dict['eng_tokenizer'],
            fre_tokenizer=data_dict['fre_tokenizer'],
            max_eng_length=data_dict['max_eng_length'],
            device=device
        )
        print(f"🇬🇧 English: {sentence}")
        print(f"🇫🇷 French:  {translation}")
        print()
    
    # Compare with old training method
    print("\n3. Comparison with old method (100% teacher forcing):")
    model_old, _, _ = train_model(
        data_file_path=None,  # Use dummy data
        epochs=4,
        batch_size=4,
        embedding_dim=64,
        lstm_units=32,
        learning_rate=0.01,
        device=device,
        use_dummy_data=True
    )
    
    print("Old method translations:")
    for sentence in ["hello world", "how are you"]:
        translation = translate_sentence_simple(
            model=model_old,
            sentence=sentence,
            eng_tokenizer=data_dict['eng_tokenizer'],
            fre_tokenizer=data_dict['fre_tokenizer'],
            max_eng_length=data_dict['max_eng_length'],
            device=device
        )
        print(f"🇬🇧 English: {sentence}")
        print(f"🇫🇷 French:  {translation}")
        print()
    
    print("✅ Demo completed successfully!")
    print("✅ Enhanced training with teacher forcing ratio implemented!")
    print("✅ Training-inference mismatch resolved!")
    
    return model, data_dict, history


def main():
    """Main function for training with real data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Neural Machine Translation Model')
    parser.add_argument('--data', type=str, default=None, help='Path to CSV file with English-French data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--lstm_units', type=int, default=256, help='LSTM hidden units')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of samples to use (None for all)')
    parser.add_argument('--demo', action='store_true', help='Run demo with dummy data')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if args.demo:
        demo()
    else:
        if args.data is None:
            print("No data file provided, running demo...")
            demo()
        else:
            print(f"Training with data file: {args.data}")
            model, data_dict, history = train_model(
                data_file_path=args.data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                embedding_dim=args.embedding_dim,
                lstm_units=args.lstm_units,
                learning_rate=args.lr,
                device=device,
                sample_size=args.sample_size
            )
            
            # Test some translations
            print("\nTesting translations:")
            test_sentences = [
                "hello world",
                "how are you today", 
                "what is your name",
                "thank you very much",
                "good morning"
            ]
            
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


def generate(sentence, model, data_dict, device='cpu'):
    """Simple generate function for easy usage"""
    return translate_sentence(
        model=model,
        sentence=sentence,
        eng_tokenizer=data_dict['eng_tokenizer'],
        fre_tokenizer=data_dict['fre_tokenizer'],
        max_eng_length=data_dict['max_eng_length'],
        device=device
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No command line args, run demo
        demo()
    else:
        main()