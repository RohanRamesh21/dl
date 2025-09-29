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

# BPE Tokenization imports
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

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
    """LSTM layer implemented from scratch using PyTorch tensors with multi-layer and bidirectional support"""
    
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_rate=0.0, 
                 batch_first=True, return_sequences=True, return_state=False, bidirectional=False, device='cpu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.bidirectional = bidirectional
        self.device = device
        
        # Effective hidden size (doubled if bidirectional)
        self.effective_hidden_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Create multiple LSTM cells for stacked layers
        self.forward_cells = []
        self.backward_cells = [] if bidirectional else None
        
        for layer in range(num_layers):
            # Input size for first layer vs subsequent layers
            if layer == 0:
                layer_input_size = input_size
            else:
                layer_input_size = self.effective_hidden_size
            
            # Forward cells
            forward_cell = LSTMCell(layer_input_size, hidden_size, device)
            self.forward_cells.append(forward_cell)
            
            # Backward cells (if bidirectional)
            if bidirectional:
                backward_cell = LSTMCell(layer_input_size, hidden_size, device)
                self.backward_cells.append(backward_cell)
        
        # For backward compatibility
        self.cells = self.forward_cells
    
    def __call__(self, x, initial_state=None):
        """Forward pass through multi-layer LSTM with bidirectional support"""
        if not self.batch_first:
            x = x.transpose(0, 1)  # Convert to batch_first
        
        batch_size, seq_len = x.size(0), x.size(1)
        
        if self.bidirectional:
            return self._bidirectional_forward(x, initial_state, batch_size, seq_len)
        else:
            return self._unidirectional_forward(x, initial_state, batch_size, seq_len)
    
    def _unidirectional_forward(self, x, initial_state, batch_size, seq_len):
        """Standard unidirectional LSTM forward pass"""
        # Initialize states for all layers
        if initial_state is None:
            h_states = [torch.zeros(batch_size, self.hidden_size, device=self.device) 
                       for _ in range(self.num_layers)]
            c_states = [torch.zeros(batch_size, self.hidden_size, device=self.device) 
                       for _ in range(self.num_layers)]
        else:
            if self.num_layers == 1:
                h_states = [initial_state[0]]
                c_states = [initial_state[1]]
            else:
                h_states = list(initial_state[0])
                c_states = list(initial_state[1])
        
        layer_outputs = []
        current_input = x
        
        # Process each time step
        for t in range(seq_len):
            layer_input = current_input[:, t, :]
            
            for layer in range(self.num_layers):
                h_t, c_t = self.forward_cells[layer](layer_input, (h_states[layer], c_states[layer]))
                h_states[layer] = h_t
                c_states[layer] = c_t
                
                if layer < self.num_layers - 1 and self.dropout_rate > 0.0:
                    layer_input = F.dropout(h_t, p=self.dropout_rate, training=True)
                else:
                    layer_input = h_t
            
            if self.return_sequences:
                layer_outputs.append(h_t.unsqueeze(1))
        
        if self.return_sequences:
            output = torch.cat(layer_outputs, dim=1)
            if not self.batch_first:
                output = output.transpose(0, 1)
        else:
            output = h_t
        
        if self.return_state:
            if self.num_layers == 1:
                return output, (h_states[0], c_states[0])
            else:
                return output, (torch.stack(h_states), torch.stack(c_states))
        else:
            return output
    
    def _bidirectional_forward(self, x, initial_state, batch_size, seq_len):
        """Bidirectional LSTM forward pass"""
        # Initialize forward and backward states
        if initial_state is None:
            # Forward states
            h_forward = [torch.zeros(batch_size, self.hidden_size, device=self.device) 
                        for _ in range(self.num_layers)]
            c_forward = [torch.zeros(batch_size, self.hidden_size, device=self.device) 
                        for _ in range(self.num_layers)]
            # Backward states
            h_backward = [torch.zeros(batch_size, self.hidden_size, device=self.device) 
                         for _ in range(self.num_layers)]
            c_backward = [torch.zeros(batch_size, self.hidden_size, device=self.device) 
                         for _ in range(self.num_layers)]
        else:
            # Handle initial states for bidirectional case
            if self.num_layers == 1:
                # Split the concatenated states
                h_init, c_init = initial_state
                h_forward = [h_init[:, :self.hidden_size]]
                c_forward = [c_init[:, :self.hidden_size]]
                h_backward = [h_init[:, self.hidden_size:]]
                c_backward = [c_init[:, self.hidden_size:]]
            else:
                h_init, c_init = initial_state
                h_forward = [h[:, :self.hidden_size] for h in h_init]
                c_forward = [c[:, :self.hidden_size] for c in c_init]
                h_backward = [h[:, self.hidden_size:] for h in h_init]
                c_backward = [c[:, self.hidden_size:] for c in c_init]
        
        layer_outputs = []
        current_input = x
        
        # Process each layer
        for layer in range(self.num_layers):
            forward_outputs = []
            backward_outputs = []
            
            # Forward direction (left to right)
            layer_h_f = h_forward[layer]
            layer_c_f = c_forward[layer]
            for t in range(seq_len):
                layer_input = current_input[:, t, :]
                layer_h_f, layer_c_f = self.forward_cells[layer](layer_input, (layer_h_f, layer_c_f))
                forward_outputs.append(layer_h_f.unsqueeze(1))
            
            # Backward direction (right to left)
            layer_h_b = h_backward[layer]
            layer_c_b = c_backward[layer]
            for t in range(seq_len - 1, -1, -1):
                layer_input = current_input[:, t, :]
                layer_h_b, layer_c_b = self.backward_cells[layer](layer_input, (layer_h_b, layer_c_b))
                backward_outputs.insert(0, layer_h_b.unsqueeze(1))  # Insert at beginning to maintain order
            
            # Concatenate forward and backward outputs
            forward_seq = torch.cat(forward_outputs, dim=1)  # (batch, seq_len, hidden)
            backward_seq = torch.cat(backward_outputs, dim=1)  # (batch, seq_len, hidden)
            bidirectional_output = torch.cat([forward_seq, backward_seq], dim=-1)  # (batch, seq_len, hidden*2)
            
            # Update states for next layer
            h_forward[layer] = layer_h_f
            c_forward[layer] = layer_c_f
            h_backward[layer] = layer_h_b
            c_backward[layer] = layer_c_b
            
            # Apply dropout between layers (not on final layer)
            if layer < self.num_layers - 1 and self.dropout_rate > 0.0:
                current_input = F.dropout(bidirectional_output, p=self.dropout_rate, training=True)
            else:
                current_input = bidirectional_output
            
            # Store outputs for the last layer if needed
            if layer == self.num_layers - 1:
                layer_outputs = bidirectional_output
        
        # Format output
        if self.return_sequences:
            output = layer_outputs
            if not self.batch_first:
                output = output.transpose(0, 1)
        else:
            output = layer_outputs[:, -1, :]  # Last timestep
        
        if self.return_state:
            # Concatenate forward and backward final states
            final_h = [torch.cat([h_f, h_b], dim=-1) for h_f, h_b in zip(h_forward, h_backward)]
            final_c = [torch.cat([c_f, c_b], dim=-1) for c_f, c_b in zip(c_forward, c_backward)]
            
            if self.num_layers == 1:
                return output, (final_h[0], final_c[0])
            else:
                return output, (torch.stack(final_h), torch.stack(final_c))
        else:
            return output
    
    def parameters(self):
        """Return parameters from all layers (forward and backward)"""
        params = []
        # Forward cells
        for cell in self.forward_cells:
            params.extend(cell.parameters())
        # Backward cells (if bidirectional)
        if self.bidirectional:
            for cell in self.backward_cells:
                params.extend(cell.parameters())
        return params


class AdditiveAttention:
    """Learnable Bahdanau-style additive attention with support for different query and key dimensions"""

    def __init__(self, query_size, key_size, attention_size=None, use_scale=True, device='cpu'):
        """
        query_size: Dimensionality of query (decoder hidden states)
        key_size: Dimensionality of key/value (encoder outputs)  
        attention_size: Size of intermediate attention projection (default = key_size)
        use_scale: Whether to scale scores by sqrt(attention_size)
        """
        self.query_size = query_size
        self.key_size = key_size
        self.attention_size = attention_size or key_size
        self.use_scale = use_scale
        self.device = device

        # Learnable weights
        self.W_q = Linear(query_size, self.attention_size, device=device)
        self.W_k = Linear(key_size, self.attention_size, device=device)
        self.v = Linear(self.attention_size, 1, device=device)

    def parameters(self):
        params = []
        params.extend(self.W_q.parameters())
        params.extend(self.W_k.parameters())
        params.extend(self.v.parameters())
        return params

    def __call__(self, query, value, key=None):
        """
        query: (batch, tgt_len, query_size)
        value: (batch, src_len, key_size)
        key:   (batch, src_len, key_size) — defaults to value
        Returns: context_vector, attention_weights
        """
        if key is None:
            key = value

        # Apply linear layers
        Wq = self.W_q(query)           # (batch, tgt_len, attention_size)
        Wk = self.W_k(key)             # (batch, src_len, attention_size)

        # Broadcast addition: Wq (B, T, A), Wk (B, S, A) → scores (B, T, S)
        # → Use broadcasting by expanding dims
        Wq_exp = Wq.unsqueeze(2)       # (B, T, 1, A)
        Wk_exp = Wk.unsqueeze(1)       # (B, 1, S, A)
        added = Wq_exp + Wk_exp        # (B, T, S, A)
        score = self.v(torch.tanh(added)).squeeze(-1)  # (B, T, S)

        if self.use_scale:
            score = score / math.sqrt(self.attention_size)

        # Softmax over source tokens (dim = 2)
        attention_weights = F.softmax(score, dim=-1)  # (B, T, S)

        # Compute context vectors: weighted sum over `value`
        context_vector = torch.bmm(attention_weights, value)  # (B, T, key_size)

        return context_vector, attention_weights


class Encoder:
    """LSTM Encoder implemented from scratch with multi-layer and bidirectional support"""
    
    def __init__(self, vocab_size, embedding_dim, lstm_units, num_layers=1, dropout_rate=0.0, bidirectional=False, device='cpu'):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.device = device
        
        # Effective output size (doubled if bidirectional)
        self.output_size = lstm_units * 2 if bidirectional else lstm_units
        
        # Components implemented from scratch
        self.embedding = Embedding(vocab_size, embedding_dim, device)
        self.lstm = LSTM(embedding_dim, lstm_units, num_layers=num_layers, 
                        dropout_rate=dropout_rate, batch_first=True, 
                        return_sequences=True, return_state=True, bidirectional=bidirectional, device=device)
    
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
    """LSTM Decoder with Attention implemented from scratch with multi-layer support"""
    
    def __init__(self, vocab_size, embedding_dim, lstm_units, encoder_output_size, num_layers=1, dropout_rate=0.0, device='cpu'):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.encoder_output_size = encoder_output_size  # Could be lstm_units*2 if encoder is bidirectional
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.device = device
        
        # Components implemented from scratch
        self.embedding = Embedding(vocab_size, embedding_dim, device)
        self.lstm = LSTM(embedding_dim, lstm_units, num_layers=num_layers,
                        dropout_rate=dropout_rate, batch_first=True,
                        return_sequences=True, return_state=True, device=device)
        
        # Attention handles different dimensions: query (decoder) vs key/value (encoder)
        # Query comes from decoder (lstm_units), Key/Value from encoder (encoder_output_size)
        self.attention = AdditiveAttention(query_size=lstm_units, key_size=encoder_output_size, use_scale=True, device=device)
        
        # Output dense: concatenates context (encoder_output_size) + decoder output (lstm_units)
        self.output_dense = Linear(encoder_output_size + lstm_units, vocab_size, device=device)
        
        # State projection layers (for bidirectional encoder compatibility)
        if encoder_output_size != lstm_units:
            self.h_projection = Linear(encoder_output_size, lstm_units, device=device)
            self.c_projection = Linear(encoder_output_size, lstm_units, device=device)
        else:
            self.h_projection = None
            self.c_projection = None
    
    def __call__(self, x, encoder_outputs, initial_state):
        """Forward pass through decoder"""
        # Project initial states if needed (for bidirectional encoder compatibility)
        if self.h_projection is not None:
            if isinstance(initial_state[0], torch.Tensor) and initial_state[0].dim() == 2:
                # Single layer case
                h_init = self.h_projection(initial_state[0])
                c_init = self.c_projection(initial_state[1])
                projected_state = (h_init, c_init)
            else:
                # Multi-layer case
                h_layers = [self.h_projection(h) for h in initial_state[0]]
                c_layers = [self.c_projection(c) for c in initial_state[1]]
                projected_state = (torch.stack(h_layers) if len(h_layers) > 1 else h_layers[0],
                                 torch.stack(c_layers) if len(c_layers) > 1 else c_layers[0])
        else:
            projected_state = initial_state
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, target_seq_len, embedding_dim)
        
        # LSTM forward pass with projected initial state
        lstm_outputs, _ = self.lstm(embedded, projected_state)
        # lstm_outputs: (batch_size, target_seq_len, lstm_units)
        
        # Apply attention mechanism (handles encoder_output_size automatically)
        context_vector, attention_weights = self.attention(
            query=lstm_outputs,      # (batch_size, target_seq_len, lstm_units)
            value=encoder_outputs,   # (batch_size, src_seq_len, encoder_output_size)
            key=encoder_outputs      # (batch_size, src_seq_len, encoder_output_size)
        )
        
        # Concatenate context vector and decoder outputs
        concatenated = torch.cat([context_vector, lstm_outputs], dim=-1)
        # concatenated: (batch_size, target_seq_len, encoder_output_size + lstm_units)
        
        # Final dense layer (TimeDistributed equivalent)
        batch_size, seq_len, concat_dim = concatenated.size()
        concatenated_flat = concatenated.reshape(batch_size * seq_len, concat_dim)
        output_flat = self.output_dense(concatenated_flat)
        decoder_outputs = output_flat.reshape(batch_size, seq_len, self.vocab_size)
        
        return decoder_outputs
    
    def step_by_step_decode(self, encoder_outputs, initial_state, target_sequence=None, 
                           teacher_forcing_ratio=1.0, max_length=50, device='cpu', sos_id=None, eos_id=None):
        """Step-by-step decoding with configurable teacher forcing ratio"""
        batch_size = encoder_outputs.size(0)
        sos_token_id = sos_id if sos_id is not None else 1  # Fallback to 1 if not provided
        eos_token_id = eos_id if eos_id is not None else 2  # Fallback to 2 if not provided
        
        # Project initial states if needed (for bidirectional encoder compatibility)
        if self.h_projection is not None:
            if isinstance(initial_state[0], torch.Tensor) and initial_state[0].dim() == 2:
                # Single layer case
                h_init = self.h_projection(initial_state[0])
                c_init = self.c_projection(initial_state[1])
                projected_state = (h_init, c_init)
            else:
                # Multi-layer case
                h_layers = [self.h_projection(h) for h in initial_state[0]]
                c_layers = [self.c_projection(c) for c in initial_state[1]]
                projected_state = (torch.stack(h_layers) if len(h_layers) > 1 else h_layers[0],
                                 torch.stack(c_layers) if len(c_layers) > 1 else c_layers[0])
        else:
            projected_state = initial_state
        
        # Initialize with SOS token
        decoder_input = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=device)
        decoder_state = projected_state
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
            # Flatten for dense layer (now handles encoder_output_size + lstm_units)
            concatenated_flat = concatenated.reshape(batch_size, self.encoder_output_size + self.lstm_units)
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
        params.extend(self.attention.parameters())
        params.extend(self.output_dense.parameters())
        # Add projection layer parameters if they exist
        if self.h_projection is not None:
            params.extend(self.h_projection.parameters())
            params.extend(self.c_projection.parameters())
        return params


class EncoderDecoderModel:
    """Complete Encoder-Decoder Model with Attention implemented from scratch with multi-layer and bidirectional support"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=256, lstm_units=256, 
                 encoder_num_layers=1, decoder_num_layers=1, dropout_rate=0.0, bidirectional=False, device='cpu'):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.device = device
        
        # Calculate encoder output size
        encoder_output_size = lstm_units * 2 if bidirectional else lstm_units
        
        # Encoder and Decoder implemented from scratch with bidirectional support
        self.encoder = Encoder(src_vocab_size, embedding_dim, lstm_units, 
                              num_layers=encoder_num_layers, dropout_rate=dropout_rate, 
                              bidirectional=bidirectional, device=device)
        self.decoder = Decoder(tgt_vocab_size, embedding_dim, lstm_units, encoder_output_size,
                              num_layers=decoder_num_layers, dropout_rate=dropout_rate, device=device)
    
    def __call__(self, encoder_inputs, decoder_inputs):
        """Forward pass through the complete model"""
        # Encoder forward pass
        encoder_outputs, state_h, state_c = self.encoder(encoder_inputs)
        
        # Use encoder final states as initial decoder states
        initial_state = (state_h, state_c)
        
        # Decoder forward pass
        decoder_outputs = self.decoder(decoder_inputs, encoder_outputs, initial_state)
        
        return decoder_outputs
    
    def forward_with_teacher_forcing(self, encoder_inputs, target_sequence, teacher_forcing_ratio=1.0, sos_id=None, eos_id=None):
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
            device=self.device,
            sos_id=sos_id,
            eos_id=eos_id
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

# BPE Tokenizer utilities
def create_and_train_bpe_tokenizer(texts, vocab_size=30000, tokenizer_path="bpe_tokenizer.json"):
    """
    Create and train a BPE tokenizer on the provided texts.
    
    Args:
        texts: List of text strings to train on
        vocab_size: Target vocabulary size (default: 30,000)
        tokenizer_path: Path to save the trained tokenizer
    
    Returns:
        Trained BPE tokenizer
    """
    print(f"🔤 Training BPE tokenizer with vocab_size={vocab_size}...")
    
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Define special tokens
    special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
    
    # Initialize trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True
    )
    
    # Train tokenizer
    tokenizer.train_from_iterator(texts, trainer)
    
    # Save tokenizer
    tokenizer.save(tokenizer_path)
    print(f"✅ BPE tokenizer trained and saved to {tokenizer_path}")
    print(f"📊 Final vocabulary size: {tokenizer.get_vocab_size()}")
    
    return tokenizer

def load_bpe_tokenizer(tokenizer_path="bpe_tokenizer.json"):
    """
    Load a previously trained BPE tokenizer.
    
    Args:
        tokenizer_path: Path to the saved tokenizer
    
    Returns:
        Loaded BPE tokenizer
    """
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        print(f"✅ Loaded BPE tokenizer from {tokenizer_path}")
        print(f"📊 Vocabulary size: {tokenizer.get_vocab_size()}")
        return tokenizer
    except FileNotFoundError:
        print(f"❌ Tokenizer file {tokenizer_path} not found!")
        return None

def get_or_create_bpe_tokenizer(texts, vocab_size=30000, tokenizer_path="bpe_tokenizer.json", force_retrain=False):
    """
    Get existing BPE tokenizer or create a new one if it doesn't exist.
    
    Args:
        texts: Training texts (used only if tokenizer doesn't exist)
        vocab_size: Target vocabulary size
        tokenizer_path: Path to save/load tokenizer
        force_retrain: Force retraining even if tokenizer exists
    
    Returns:
        BPE tokenizer instance
    """
    if not force_retrain and os.path.exists(tokenizer_path):
        tokenizer = load_bpe_tokenizer(tokenizer_path)
        if tokenizer is not None:
            return tokenizer
    
    print(f"🔄 Creating new BPE tokenizer...")
    return create_and_train_bpe_tokenizer(texts, vocab_size, tokenizer_path)


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


def train_step(model, encoder_inputs, decoder_inputs, targets, optimizer, teacher_forcing_ratio=1.0, clip_grad_norm=1.0, sos_id=None, eos_id=None):
    """Single training step with teacher forcing ratio and gradient clipping"""
    optimizer.zero_grad()
    
    # Forward pass with teacher forcing
    if teacher_forcing_ratio < 1.0:
        # Use step-by-step decoding with teacher forcing ratio
        target_sequence = torch.cat([decoder_inputs, targets[:, -1:]], dim=1)  # Add last target token
        predictions = model.forward_with_teacher_forcing(
            encoder_inputs, target_sequence, teacher_forcing_ratio, sos_id=sos_id, eos_id=eos_id
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
    
    # Create BPE tokenizers
    print("🔤 Setting up BPE tokenizers...")
    
    # Combine training texts for tokenizer training
    combined_texts = list(eng_train) + list(fre_train)
    
    # Create or load BPE tokenizer (shared for both English and French)
    bpe_tokenizer = get_or_create_bpe_tokenizer(
        texts=combined_texts,
        vocab_size=30000,
        tokenizer_path="bpe_tokenizer.json"
    )
    
    # Convert to sequences using BPE tokenizer
    print("🔤 Tokenizing sequences...")
    eng_train_seq = [bpe_tokenizer.encode(text).ids for text in eng_train]
    eng_val_seq = [bpe_tokenizer.encode(text).ids for text in eng_val]
    fre_train_seq = [bpe_tokenizer.encode(text).ids for text in fre_train]
    fre_val_seq = [bpe_tokenizer.encode(text).ids for text in fre_val]
    
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
    
    # Extract special token IDs from BPE tokenizer - handle mixed format
    pad_id = bpe_tokenizer.token_to_id("[PAD]")
    unk_id = bpe_tokenizer.token_to_id("[UNK]")
    
    # For SOS and EOS, the training data uses 'sos' and 'eos', not symbolic tokens
    sos_id = bpe_tokenizer.token_to_id("sos")
    eos_id = bpe_tokenizer.token_to_id("eos")
    
    # Fallback to symbolic if not found
    if sos_id is None:
        sos_id = bpe_tokenizer.token_to_id("[SOS]")
    if eos_id is None:
        eos_id = bpe_tokenizer.token_to_id("[EOS]")
    
    # Final fallbacks based on known vocab structure
    if sos_id is None:
        sos_id = 36  # From vocab: "sos": 36
    if eos_id is None:
        eos_id = 35  # From vocab: "eos": 35
    
    print(f"📊 Special token IDs: PAD={pad_id}, UNK={unk_id}, SOS={sos_id}, EOS={eos_id}")
    print(f"📊 Using textual tokens: sos={sos_id}, eos={eos_id} (not symbolic [SOS]/[EOS])")
    
    return {
        'eng_train_pad': eng_train_pad,
        'eng_val_pad': eng_val_pad,
        'fre_train_pad': fre_train_pad,
        'fre_val_pad': fre_val_pad,
        'bpe_tokenizer': bpe_tokenizer,  # Single tokenizer for both languages
        'vocab_size': bpe_tokenizer.get_vocab_size(),  # Same vocab size for both
        'max_eng_length': max_eng_length,
        'max_fre_length': max_fre_length,
        'pad_id': pad_id,
        'unk_id': unk_id,
        'sos_id': sos_id,
        'eos_id': eos_id
    }


def train_model_enhanced(data_file_path=None, epochs=10, batch_size=64, embedding_dim=256,
                        lstm_units=256, learning_rate=0.001, device='cpu', sample_size=None, 
                        use_dummy_data=False, teacher_forcing_schedule='linear',
                        encoder_num_layers=1, decoder_num_layers=1, dropout_rate=0.0, bidirectional=False):
    """Enhanced training pipeline with teacher forcing scheduling"""
    print("=" * 60)
    print("ENHANCED NEURAL MACHINE TRANSLATION TRAINING")
    print("With Teacher Forcing Ratio Scheduling")
    print("=" * 60)
    
    print("Loading and preprocessing data...")
    data_dict = prepare_data(data_file_path, sample_size, use_dummy_data)
    
    print(f"Vocabulary size: {data_dict['vocab_size']}")
    print(f"Special tokens - SOS: {data_dict['sos_id']}, EOS: {data_dict['eos_id']}")
    
    # Create model
    model = EncoderDecoderModel(
        src_vocab_size=data_dict['vocab_size'],
        tgt_vocab_size=data_dict['vocab_size'],
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        encoder_num_layers=encoder_num_layers,
        decoder_num_layers=decoder_num_layers,
        dropout_rate=dropout_rate,
        bidirectional=bidirectional,
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
    print(f"Encoder: {'Bidirectional' if bidirectional else 'Unidirectional'} LSTM")
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
                optimizer, teacher_forcing_ratio=tf_ratio,
                sos_id=data_dict['sos_id'], eos_id=data_dict['eos_id']
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


def translate_sentence(model, sentence, bpe_tokenizer, max_eng_length, device='cpu', max_output_length=30, temperature=0.7):
    """Translation function using BPE tokenizer"""
    model.train = False
    
    # Tokenize and pad input using BPE tokenizer
    encoded = bpe_tokenizer.encode(sentence)
    sequence = encoded.ids
    
    # Pad sequence manually
    if len(sequence) > max_eng_length:
        sequence = sequence[:max_eng_length]
    else:
        pad_id = bpe_tokenizer.token_to_id("[PAD]")
        sequence = sequence + [pad_id] * (max_eng_length - len(sequence))
    
    encoder_inputs = torch.tensor([sequence], dtype=torch.long, device=device)
    
    # Get special tokens - handle mixed token format in BPE tokenizer
    # The training data uses 'sos' and 'eos' tokens, not '[SOS]' and '[EOS]'
    sos_token_id = bpe_tokenizer.token_to_id("sos")
    eos_token_id = bpe_tokenizer.token_to_id("eos")
    
    # Fallback to symbolic tokens if textual ones don't exist
    if sos_token_id is None:
        sos_token_id = bpe_tokenizer.token_to_id("[SOS]")
    if eos_token_id is None:
        eos_token_id = bpe_tokenizer.token_to_id("[EOS]")
        
    # Final fallback to default IDs based on vocab structure
    if sos_token_id is None:
        sos_token_id = 36  # From vocab: "sos": 36
    if eos_token_id is None:
        eos_token_id = 35   # From vocab: "eos": 35
    
    # Encode input
    encoder_outputs, state_h, state_c = model.encoder(encoder_inputs)
    initial_state = (state_h, state_c)
    
    # Project initial states if needed (for bidirectional encoder compatibility)
    if model.decoder.h_projection is not None:
        if isinstance(initial_state[0], torch.Tensor) and initial_state[0].dim() == 2:
            # Single layer case
            h_init = model.decoder.h_projection(initial_state[0])
            c_init = model.decoder.c_projection(initial_state[1])
            decoder_state = (h_init, c_init)
        else:
            # Multi-layer case
            h_layers = [model.decoder.h_projection(h) for h in initial_state[0]]
            c_layers = [model.decoder.c_projection(c) for c in initial_state[1]]
            decoder_state = (torch.stack(h_layers) if len(h_layers) > 1 else h_layers[0],
                           torch.stack(c_layers) if len(c_layers) > 1 else c_layers[0])
    else:
        decoder_state = initial_state
    
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
                value=encoder_outputs,  # (1, src_seq_len, encoder_output_size)
                key=encoder_outputs     # (1, src_seq_len, encoder_output_size)
            )
            
            # Concatenate and get predictions (matching training)
            concatenated = torch.cat([context_vector, lstm_output], dim=-1)
            # Use the correct size: encoder_output_size + lstm_units
            concatenated_flat = concatenated.reshape(1, model.decoder.encoder_output_size + model.decoder.lstm_units)
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
    
    # Convert tokens to text using BPE tokenizer with cleaning
    if not generated_tokens:
        return "<empty>"
    
    # Filter out special tokens for cleaner output
    filtered_tokens = []
    special_tokens = {0, 1, 2, 3, 35, 36}  # PAD, UNK, [SOS], [EOS], eos, sos
    
    for token_id in generated_tokens:
        # Skip special tokens for cleaner output
        if token_id not in special_tokens:
            filtered_tokens.append(token_id)
    
    if filtered_tokens:
        translation = bpe_tokenizer.decode(filtered_tokens)
    else:
        # If all tokens were special, decode everything
        translation = bpe_tokenizer.decode(generated_tokens)
    
    # Clean up the translation
    translation = translation.strip()
    # Remove any remaining special token text
    for special_text in ['[SOS]', '[EOS]', '[PAD]', '[UNK]', 'sos', 'eos']:
        translation = translation.replace(special_text, '')
    translation = translation.strip()
    
    return translation if translation else "<no translation>"


def generate(sentence, model, data_dict, device='cpu'):
    """Simple generate function for easy usage with BPE tokenizer"""
    return translate_sentence(
        model=model,
        sentence=sentence,
        bpe_tokenizer=data_dict['bpe_tokenizer'],
        max_eng_length=data_dict['max_eng_length'],
        device=device
    )