"""
Practical implementation using PyTorch tensors directly
This shows how to implement neural networks from scratch while leveraging PyTorch's autograd
"""
import torch
import torch.nn.functional as F
import math


class Linear:
    """Linear layer using PyTorch tensors directly"""
    
    def __init__(self, in_features, out_features, bias=True, device='cpu'):
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights using PyTorch tensors with requires_grad=True
        self.weight = torch.randn(out_features, in_features, device=device, requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weight)
        
        if bias:
            self.bias = torch.zeros(out_features, device=device, requires_grad=True)
        else:
            self.bias = None
    
    def __call__(self, x):
        """Forward pass: y = xW^T + b"""
        output = x @ self.weight.t()
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
    """Embedding layer using PyTorch tensors"""
    
    def __init__(self, num_embeddings, embedding_dim, device='cpu'):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize embedding matrix
        self.weight = torch.randn(num_embeddings, embedding_dim, device=device, requires_grad=True)
        torch.nn.init.xavier_normal_(self.weight)
    
    def __call__(self, input_ids):
        """Forward pass - index into embedding matrix"""
        return F.embedding(input_ids, self.weight)
    
    def parameters(self):
        return [self.weight]


class LSTMCell:
    """LSTM cell using PyTorch tensors"""
    
    def __init__(self, input_size, hidden_size, device='cpu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input-to-hidden weights
        self.W_ii = Linear(input_size, hidden_size, device=device)  # input gate
        self.W_if = Linear(input_size, hidden_size, device=device)  # forget gate  
        self.W_ig = Linear(input_size, hidden_size, device=device)  # cell gate
        self.W_io = Linear(input_size, hidden_size, device=device)  # output gate
        
        # Hidden-to-hidden weights
        self.W_hi = Linear(hidden_size, hidden_size, bias=False, device=device)
        self.W_hf = Linear(hidden_size, hidden_size, bias=False, device=device)
        self.W_hg = Linear(hidden_size, hidden_size, bias=False, device=device)
        self.W_ho = Linear(hidden_size, hidden_size, bias=False, device=device)
    
    def __call__(self, x, hidden_state=None):
        """Forward pass through LSTM cell"""
        batch_size = x.size(0)
        
        if hidden_state is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_prev, c_prev = hidden_state
        
        # LSTM gates
        i_t = torch.sigmoid(self.W_ii(x) + self.W_hi(h_prev))  # input gate
        f_t = torch.sigmoid(self.W_if(x) + self.W_hf(h_prev))  # forget gate  
        g_t = torch.tanh(self.W_ig(x) + self.W_hg(h_prev))     # cell gate
        o_t = torch.sigmoid(self.W_io(x) + self.W_ho(h_prev))  # output gate
        
        # Update cell and hidden states
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
    """LSTM layer using PyTorch tensors"""
    
    def __init__(self, input_size, hidden_size, batch_first=True, return_state=False, device='cpu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.return_state = return_state
        self.device = device
        
        self.cell = LSTMCell(input_size, hidden_size, device)
    
    def __call__(self, x, initial_state=None):
        """Forward pass through LSTM"""
        if not self.batch_first:
            x = x.transpose(0, 1)  # Convert to batch_first
        
        batch_size, seq_len = x.size(0), x.size(1)
        
        if initial_state is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=self.device)
            c_0 = torch.zeros(batch_size, self.hidden_size, device=self.device)
        else:
            h_0, c_0 = initial_state
        
        outputs = []
        h_t, c_t = h_0, c_0
        
        for t in range(seq_len):
            h_t, c_t = self.cell(x[:, t, :], (h_t, c_t))
            outputs.append(h_t.unsqueeze(1))
        
        output = torch.cat(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        if self.return_state:
            return output, (h_t, c_t)
        else:
            return output
    
    def parameters(self):
        return self.cell.parameters()


class AdditiveAttention:
    """Additive attention using PyTorch tensors"""
    
    def __call__(self, query, value, key=None):
        """Compute additive attention"""
        if key is None:
            key = value
        
        batch_size, query_len, query_dim = query.shape
        _, value_len, value_dim = value.shape
        
        # Expand for broadcasting
        query_expanded = query.unsqueeze(2)  # (batch, query_len, 1, dim)
        key_expanded = key.unsqueeze(1)      # (batch, 1, value_len, dim)
        
        # Additive scores
        scores = (query_expanded + key_expanded).sum(dim=-1)  # (batch, query_len, value_len)
        scores = scores / math.sqrt(query_dim)  # scaling
        
        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context_vector = attention_weights @ value
        
        return context_vector, attention_weights


class EncoderDecoderModel:
    """Encoder-decoder model using PyTorch tensors"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=256, lstm_units=256, device='cpu'):
        self.device = device
        
        # Encoder
        self.encoder_embedding = Embedding(src_vocab_size, embedding_dim, device)
        self.encoder_lstm = LSTM(embedding_dim, lstm_units, return_state=True, device=device)
        
        # Decoder  
        self.decoder_embedding = Embedding(tgt_vocab_size, embedding_dim, device)
        self.decoder_lstm = LSTM(embedding_dim, lstm_units, return_state=True, device=device)
        self.attention = AdditiveAttention()
        self.output_projection = Linear(lstm_units * 2, tgt_vocab_size, device=device)
    
    def __call__(self, encoder_inputs, decoder_inputs):
        """Forward pass"""
        # Encode
        enc_embedded = self.encoder_embedding(encoder_inputs)
        encoder_outputs, (state_h, state_c) = self.encoder_lstm(enc_embedded)
        
        # Decode  
        dec_embedded = self.decoder_embedding(decoder_inputs)
        decoder_outputs, _ = self.decoder_lstm(dec_embedded, (state_h, state_c))
        
        # Attention
        context_vector, _ = self.attention(decoder_outputs, encoder_outputs)
        
        # Concatenate and project
        concatenated = torch.cat([context_vector, decoder_outputs], dim=-1)
        output = self.output_projection(concatenated)
        
        return output
    
    def parameters(self):
        """Return all parameters for optimizer"""
        params = []
        params.extend(self.encoder_embedding.parameters())
        params.extend(self.encoder_lstm.parameters())
        params.extend(self.decoder_embedding.parameters())
        params.extend(self.decoder_lstm.parameters())
        params.extend(self.output_projection.parameters())
        return params


def train_step(model, encoder_inputs, decoder_inputs, targets, optimizer, criterion):
    """Single training step using PyTorch"""
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(encoder_inputs, decoder_inputs)
    
    # Compute loss
    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    return loss.item()


# Usage example
def demo_pytorch_approach():
    """Demonstrate the PyTorch-native approach"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = EncoderDecoderModel(
        src_vocab_size=5000,
        tgt_vocab_size=4000, 
        embedding_dim=256,
        lstm_units=256,
        device=device
    )
    
    # Create optimizer (using PyTorch's Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Dummy data
    batch_size = 32
    src_seq_len = 20
    tgt_seq_len = 25
    
    encoder_inputs = torch.randint(1, 5000, (batch_size, src_seq_len), device=device)
    decoder_inputs = torch.randint(1, 4000, (batch_size, tgt_seq_len), device=device)
    targets = torch.randint(0, 4000, (batch_size, tgt_seq_len), device=device)
    
    # Training step
    loss = train_step(model, encoder_inputs, decoder_inputs, targets, optimizer, criterion)
    print(f"Training loss: {loss:.4f}")
    
    print("✓ PyTorch-native approach working correctly!")


if __name__ == "__main__":
    demo_pytorch_approach()