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
        Single decoding step (for inference) with proper state tracking
        
        Args:
            decoder_input: Current decoder input token (batch_size, 1)
            encoder_outputs: Encoder outputs (batch_size, src_seq_len, lstm_units)
            state: Current decoder state tuple (state_h, state_c)
            
        Returns:
            output: Output logits (batch_size, 1, tgt_vocab_size)
            new_state: New decoder state tuple (state_h, state_c)
        """
        # Embed input
        embedded = self.decoder.embedding(decoder_input)
        
        # LSTM step with proper state update
        lstm_output, new_states = self.decoder.lstm(embedded, initial_state=[state])
        new_state = new_states[0]  # Extract (h, c) from list
        
        # Apply attention
        context_vector, _ = self.decoder.attention(
            query=lstm_output,
            value=encoder_outputs,
            key=encoder_outputs
        )
        
        # Concatenate context vector and LSTM output
        concatenated = torch.cat([context_vector.data, lstm_output.data], dim=-1)
        concatenated = Tensor(concatenated, requires_grad=(context_vector.requires_grad or lstm_output.requires_grad))
        
        # Apply output dense layer
        batch_size, seq_len, concat_dim = concatenated.shape
        concatenated_flat = concatenated.view(batch_size * seq_len, concat_dim)
        output_flat = self.decoder.output_dense(concatenated_flat)
        output = output_flat.view(batch_size, seq_len, self.tgt_vocab_size)
        
        return output, new_state
    
    def generate(self, encoder_inputs, max_length=50, start_token_id=1, end_token_id=2):
        """
        Generate target sequence using greedy decoding with improved EOS handling
        
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
        decoder_input = Tensor(decoder_input.float())
        
        generated_tokens = []
        state = (state_h, state_c)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for step in range(max_length):
            # Single decoding step
            output, state = self.decode_step(decoder_input, encoder_outputs, state)
            
            # Get predicted token (greedy)
            predicted_token = output.data.argmax(dim=-1)  # (batch_size, 1)
            
            # Mask tokens for finished sequences (prevent generation after EOS)
            predicted_token = predicted_token.masked_fill(finished.unsqueeze(1), end_token_id)
            generated_tokens.append(predicted_token)
            
            # Update finished sequences
            finished = finished | (predicted_token.squeeze(1) == end_token_id)
            
            # Break if all sequences are finished
            if finished.all():
                break
            
            # Use predicted token as next input (convert to float for embedding)
            decoder_input = Tensor(predicted_token.float())
        
        # Concatenate generated tokens
        if generated_tokens:
            generated_sequence = torch.cat(generated_tokens, dim=1)
        else:
            generated_sequence = torch.empty((batch_size, 0), dtype=torch.long, device=self.device)
        
        return generated_sequence
    
    def beam_search(self, encoder_inputs, beam_width=3, max_length=50, start_token_id=1, end_token_id=2, length_penalty=0.6):
        """
        Generate target sequence using beam search decoding
        
        Args:
            encoder_inputs: Source sequence tokens (batch_size, src_seq_len)
            beam_width: Number of beams to maintain
            max_length: Maximum generation length
            start_token_id: Start of sequence token ID
            end_token_id: End of sequence token ID
            length_penalty: Length normalization penalty (0.0-1.0, higher = prefer longer sequences)
            
        Returns:
            best_sequences: Best generated sequences (batch_size, seq_len)
        """
        batch_size = encoder_inputs.shape[0]
        
        # For simplicity, only handle batch_size=1 in beam search
        if batch_size != 1:
            # Fall back to greedy for batch processing
            return self.generate(encoder_inputs, max_length, start_token_id, end_token_id)
        
        # Encode input
        encoder_outputs, state_h, state_c = self.encode(encoder_inputs)
        
        # Initialize beams: (score, sequence, state)
        beams = [(0.0, [start_token_id], (state_h, state_c))]
        completed_sequences = []
        
        for step in range(max_length):
            new_beams = []
            
            for score, sequence, state in beams:
                if sequence[-1] == end_token_id:
                    # Apply length penalty and add to completed
                    length_penalty_factor = ((5 + len(sequence)) / 6) ** length_penalty
                    normalized_score = score / length_penalty_factor
                    completed_sequences.append((normalized_score, sequence))
                    continue
                
                # Prepare input for this beam
                decoder_input = torch.tensor([[sequence[-1]]], dtype=torch.float32, device=self.device)
                decoder_input = Tensor(decoder_input)
                
                # Get predictions for this step
                output, new_state = self.decode_step(decoder_input, encoder_outputs, state)
                log_probs = torch.log_softmax(output.data, dim=-1).squeeze()
                
                # Get top-k candidates
                top_scores, top_indices = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    token_score = top_scores[i].item()
                    token_id = top_indices[i].item()
                    new_score = score + token_score
                    new_sequence = sequence + [token_id]
                    new_beams.append((new_score, new_sequence, new_state))
            
            # Keep only top beam_width beams
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
            
            # Early stopping if we have enough completed sequences
            if len(completed_sequences) >= beam_width:
                break
        
        # Add remaining beams to completed sequences
        for score, sequence, _ in beams:
            length_penalty_factor = ((5 + len(sequence)) / 6) ** length_penalty
            normalized_score = score / length_penalty_factor
            completed_sequences.append((normalized_score, sequence))
        
        # Return best sequence
        if completed_sequences:
            best_sequence = max(completed_sequences, key=lambda x: x[0])[1][1:]  # Remove start token
            return torch.tensor([best_sequence], dtype=torch.long, device=self.device)
        else:
            # Fallback to greedy if no completed sequences
            return self.generate(encoder_inputs, max_length, start_token_id, end_token_id)
    
    def forward_with_scheduled_sampling(self, encoder_inputs, decoder_inputs, targets, teacher_forcing_ratio=1.0):
        """
        Forward pass with scheduled sampling during training
        
        Args:
            encoder_inputs: Source sequence tokens (batch_size, src_seq_len)
            decoder_inputs: Target input sequence tokens (batch_size, tgt_seq_len)
            targets: Target output sequence tokens (batch_size, tgt_seq_len)
            teacher_forcing_ratio: Probability of using teacher forcing (0.0-1.0)
            
        Returns:
            decoder_outputs: Output logits (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        if teacher_forcing_ratio == 1.0:
            # Full teacher forcing (current implementation)
            return self.forward(encoder_inputs, decoder_inputs)
        
        batch_size, target_len = decoder_inputs.shape
        
        # Encode input
        encoder_outputs, state_h, state_c = self.encode(encoder_inputs)
        state = (state_h, state_c)
        
        # Initialize outputs list
        outputs = []
        
        # First input is always from ground truth (SOS token)
        decoder_input = decoder_inputs[:, :1]  # (batch_size, 1)
        
        for t in range(target_len):
            # Single decoding step
            output, state = self.decode_step(decoder_input, encoder_outputs, state)
            outputs.append(output)
            
            # Decide next input based on teacher forcing ratio
            if t < target_len - 1:  # Don't need next input for last step
                use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
                
                if use_teacher_forcing:
                    # Use ground truth token
                    decoder_input = decoder_inputs[:, t+1:t+2]
                else:
                    # Use model's own prediction
                    predicted_token = output.data.argmax(dim=-1)
                    decoder_input = Tensor(predicted_token.float())
        
        # Concatenate all outputs
        final_output = torch.cat([out.data for out in outputs], dim=1)
        return Tensor(final_output, requires_grad=True)


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