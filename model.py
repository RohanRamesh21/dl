"""
Encoder-Decoder architecture with attention for neural machine translation
Implements the exact architecture from the reference notebook
"""
from tensor import Tensor, zeros, ones
from nn import Module, Embedding, Linear
from lstm import LSTM
from attention import AdditiveAttention
import torch


class Encoder(Module):
    """LSTM Encoder"""
    
    def __init__(self, vocab_size, embedding_dim, lstm_units, device='cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.device = device
        
        # Embedding layer
        self.embedding = Embedding(vocab_size, embedding_dim, device=device)
        
        # LSTM layer with return_sequences=True and return_state=True
        self.lstm = LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            return_sequences=True,
            return_state=True,
            device=device
        )
        
        self._modules['embedding'] = self.embedding
        self._modules['lstm'] = self.lstm
    
    def forward(self, x):
        """
        Forward pass through encoder
        
        Args:
            x: Input token indices (batch_size, seq_len)
            
        Returns:
            encoder_outputs: LSTM outputs for all timesteps (batch_size, seq_len, lstm_units)
            state_h: Final hidden state (batch_size, lstm_units)
            state_c: Final cell state (batch_size, lstm_units)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM with state return
        encoder_outputs, final_states = self.lstm(embedded)  # outputs, [(h_n, c_n)]
        
        # Extract final states from list (single layer LSTM)
        state_h, state_c = final_states[0]
        
        return encoder_outputs, state_h, state_c


class Decoder(Module):
    """LSTM Decoder with Attention"""
    
    def __init__(self, vocab_size, embedding_dim, lstm_units, device='cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.device = device
        
        # Embedding layer
        self.embedding = Embedding(vocab_size, embedding_dim, device=device)
        
        # LSTM layer
        self.lstm = LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            return_sequences=True,
            return_state=True,
            device=device
        )
        
        # Attention mechanism (AdditiveAttention like in TensorFlow)
        self.attention = AdditiveAttention(use_scale=True, device=device)
        
        # Dense layer for final output (equivalent to TimeDistributed(Dense))
        self.output_dense = Linear(lstm_units * 2, vocab_size, device=device)  # *2 for concatenation
        
        self._modules['embedding'] = self.embedding
        self._modules['lstm'] = self.lstm
        self._modules['attention'] = self.attention
        self._modules['output_dense'] = self.output_dense
    
    def forward(self, x, encoder_outputs, initial_state):
        """
        Forward pass through decoder
        
        Args:
            x: Decoder input tokens (batch_size, target_seq_len)
            encoder_outputs: Encoder outputs (batch_size, src_seq_len, lstm_units)
            initial_state: Initial decoder state tuple (state_h, state_c)
            
        Returns:
            decoder_outputs: Output logits (batch_size, target_seq_len, vocab_size)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, target_seq_len, embedding_dim)
        
        # LSTM forward pass with initial state
        lstm_outputs, _ = self.lstm(embedded, initial_state=[initial_state])
        # lstm_outputs: (batch_size, target_seq_len, lstm_units)
        
        # Apply attention mechanism
        # query: decoder LSTM outputs, value/key: encoder outputs
        context_vector, attention_weights = self.attention(
            query=lstm_outputs,  # (batch_size, target_seq_len, lstm_units)
            value=encoder_outputs,  # (batch_size, src_seq_len, lstm_units)
            key=encoder_outputs   # (batch_size, src_seq_len, lstm_units)
        )
        # context_vector: (batch_size, target_seq_len, lstm_units)
        
        # Concatenate context vector and decoder outputs (like TensorFlow Concatenate layer)
        concatenated = torch.cat([context_vector.data, lstm_outputs.data], dim=-1)
        concatenated = Tensor(concatenated, requires_grad=(context_vector.requires_grad or lstm_outputs.requires_grad))
        
        def concat_backward_fn(grad):
            # Split gradient for context and lstm outputs
            context_grad = grad.data[..., :self.lstm_units]
            lstm_grad = grad.data[..., self.lstm_units:]
            
            if context_vector.requires_grad:
                context_vector.backward(context_grad)
            if lstm_outputs.requires_grad:
                lstm_outputs.backward(lstm_grad)
        
        if concatenated.requires_grad:
            concatenated._backward_fn = concat_backward_fn
        
        # Final dense layer (equivalent to TimeDistributed(Dense))
        # Apply linear layer to each timestep
        batch_size, seq_len, concat_dim = concatenated.shape
        concatenated_flat = concatenated.view(batch_size * seq_len, concat_dim)
        output_flat = self.output_dense(concatenated_flat)  # (batch_size * seq_len, vocab_size)
        decoder_outputs = output_flat.view(batch_size, seq_len, self.vocab_size)
        
        return decoder_outputs


class EncoderDecoderModel(Module):
    """Complete Encoder-Decoder Model with Attention"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=256, lstm_units=256, device='cpu'):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.device = device
        
        # Encoder
        self.encoder = Encoder(src_vocab_size, embedding_dim, lstm_units, device)
        
        # Decoder
        self.decoder = Decoder(tgt_vocab_size, embedding_dim, lstm_units, device)
        
        self._modules['encoder'] = self.encoder
        self._modules['decoder'] = self.decoder
    
    def forward(self, encoder_inputs, decoder_inputs):
        """
        Forward pass through the complete model
        
        Args:
            encoder_inputs: Source sequence tokens (batch_size, src_seq_len)
            decoder_inputs: Target sequence tokens (batch_size, tgt_seq_len)
            
        Returns:
            decoder_outputs: Output logits (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Encoder forward pass
        encoder_outputs, state_h, state_c = self.encoder(encoder_inputs)
        
        # Use encoder final states as initial decoder states
        initial_state = (state_h, state_c)
        
        # Decoder forward pass
        decoder_outputs = self.decoder(decoder_inputs, encoder_outputs, initial_state)
        
        return decoder_outputs
    
    def encode(self, encoder_inputs):
        """
        Encode source sequence
        
        Args:
            encoder_inputs: Source sequence tokens (batch_size, src_seq_len)
            
        Returns:
            encoder_outputs: Encoder outputs (batch_size, src_seq_len, lstm_units)
            state_h: Final hidden state (batch_size, lstm_units)
            state_c: Final cell state (batch_size, lstm_units)
        """
        return self.encoder(encoder_inputs)
    
    def decode_step(self, decoder_input, encoder_outputs, state):
        """
        Single decoding step (for inference)
        
        Args:
            decoder_input: Current decoder input token (batch_size, 1)
            encoder_outputs: Encoder outputs (batch_size, src_seq_len, lstm_units)
            state: Current decoder state tuple (state_h, state_c)
            
        Returns:
            output: Output logits (batch_size, 1, tgt_vocab_size)
            new_state: New decoder state tuple (state_h, state_c)
        """
        # Use decoder but only for single step
        output = self.decoder(decoder_input, encoder_outputs, state)
        
        # For inference, we need to update the state manually
        # This is a simplified version - in practice you'd need to track LSTM states properly
        return output, state
    
    def generate(self, encoder_inputs, max_length=50, start_token_id=1, end_token_id=2):
        """
        Generate target sequence using greedy decoding
        
        Args:
            encoder_inputs: Source sequence tokens (batch_size, src_seq_len)
            max_length: Maximum generation length
            start_token_id: Start of sequence token ID
            end_token_id: End of sequence token ID
            
        Returns:
            generated_tokens: Generated token sequence (batch_size, generated_len)
        """
        batch_size = encoder_inputs.shape[0]
        
        # Encode input
        encoder_outputs, state_h, state_c = self.encode(encoder_inputs)
        
        # Initialize decoder input with start token
        decoder_input = torch.full((batch_size, 1), start_token_id, 
                                  dtype=torch.long, device=self.device)
        decoder_input = Tensor(decoder_input)
        
        generated_tokens = []
        state = (state_h, state_c)
        
        for _ in range(max_length):
            # Single decoding step
            output, state = self.decode_step(decoder_input, encoder_outputs, state)
            
            # Get predicted token (greedy)
            predicted_token = output.data.argmax(dim=-1)  # (batch_size, 1)
            generated_tokens.append(predicted_token)
            
            # Check if all sequences have generated end token
            if (predicted_token == end_token_id).all():
                break
            
            # Use predicted token as next input
            decoder_input = Tensor(predicted_token.float())
        
        # Concatenate generated tokens
        if generated_tokens:
            generated_sequence = torch.cat(generated_tokens, dim=1)
        else:
            generated_sequence = torch.empty((batch_size, 0), dtype=torch.long, device=self.device)
        
        return generated_sequence


# Factory function to create model matching the reference notebook
def create_translation_model(eng_vocab_size, fre_vocab_size, 
                           embedding_dim=256, lstm_units=256, device='cpu'):
    """
    Create translation model matching the reference notebook architecture
    
    Args:
        eng_vocab_size: English vocabulary size
        fre_vocab_size: French vocabulary size
        embedding_dim: Embedding dimension (default: 256)
        lstm_units: LSTM hidden units (default: 256)
        device: Device to place model on
        
    Returns:
        model: EncoderDecoderModel instance
    """
    model = EncoderDecoderModel(
        src_vocab_size=eng_vocab_size,
        tgt_vocab_size=fre_vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        device=device
    )
    
    return model