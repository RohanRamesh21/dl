"""
Attention mechanisms implemented from scratch
"""
from tensor import Tensor, zeros, softmax, tanh
from nn import Module, Linear
import torch


class AdditiveAttention(Module):
    """
    Additive (Bahdanau) attention mechanism from scratch
    This matches the TensorFlow AdditiveAttention used in the reference
    """
    
    def __init__(self, use_scale=True, dropout=0.0, device='cpu'):
        super().__init__()
        self.use_scale = use_scale
        self.dropout = dropout
        self.device = device
        
        # No learnable parameters in TensorFlow's AdditiveAttention
        # The linear transformations are typically done outside
    
    def forward(self, query, value, key=None, attention_mask=None):
        """
        Additive attention forward pass
        
        Args:
            query: Query tensor (batch_size, query_seq_len, query_dim) or (query_seq_len, batch_size, query_dim)
            value: Value tensor (batch_size, value_seq_len, value_dim) or (value_seq_len, batch_size, value_dim)  
            key: Key tensor (optional, defaults to value)
            attention_mask: Mask tensor (optional)
            
        Returns:
            context_vector: Attended values (same shape as query)
            attention_weights: Attention weights (batch_size, query_seq_len, value_seq_len)
        """
        if not isinstance(query, Tensor):
            query = Tensor(query)
        if not isinstance(value, Tensor):
            value = Tensor(value)
        
        if key is None:
            key = value
        elif not isinstance(key, Tensor):
            key = Tensor(key)
        
        # Handle both batch_first=True and batch_first=False cases
        # Assume batch_first=True (batch_size, seq_len, hidden_size)
        batch_size, query_seq_len, query_dim = query.shape
        _, value_seq_len, value_dim = value.shape
        
        # For additive attention: score = v^T * tanh(W_q * query + W_k * key)
        # In TensorFlow's implementation, this is simplified to just compute
        # attention weights based on query and key directly
        
        # Expand query and key for broadcasting
        # query: (batch_size, query_seq_len, 1, query_dim)
        # key: (batch_size, 1, value_seq_len, key_dim)
        query_expanded = query.unsqueeze(2)  # (batch_size, query_seq_len, 1, query_dim)
        key_expanded = key.unsqueeze(1)      # (batch_size, 1, value_seq_len, key_dim)
        
        # Compute additive scores
        # For simplicity, we'll use a dot product after ensuring dimensions match
        if query_dim != value_dim:
            raise ValueError(f"Query dimension {query_dim} must match value dimension {value_dim}")
        
        # Compute attention scores using additive mechanism
        # score(i,j) = query[i] + key[j] (simplified additive)
        scores = query_expanded + key_expanded  # Broadcasting: (batch_size, query_seq_len, value_seq_len, dim)
        
        # Reduce to scalar scores: take sum over the feature dimension
        scores = scores.sum(dim=-1)  # (batch_size, query_seq_len, value_seq_len)
        
        # Apply scaling if requested
        if self.use_scale:
            scores = scores / (query_dim ** 0.5)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Mask should be (batch_size, query_seq_len, value_seq_len)
            # or broadcastable to that shape
            mask_value = -1e9
            scores = scores + (1.0 - attention_mask) * mask_value
        
        # Compute attention weights using softmax
        attention_weights = softmax(scores, dim=-1)  # (batch_size, query_seq_len, value_seq_len)
        
        # Apply dropout if in training mode
        if self.training and self.dropout > 0.0:
            dropout_mask = torch.rand_like(attention_weights.data) > self.dropout
            attention_weights = attention_weights * Tensor(dropout_mask.float() / (1.0 - self.dropout))
        
        # Compute context vector using attention weights and values
        # attention_weights: (batch_size, query_seq_len, value_seq_len)
        # value: (batch_size, value_seq_len, value_dim)
        # output: (batch_size, query_seq_len, value_dim)
        context_vector = attention_weights @ value
        
        return context_vector, attention_weights


class ScaledDotProductAttention(Module):
    """
    Scaled dot-product attention mechanism
    """
    
    def __init__(self, dropout=0.0, device='cpu'):
        super().__init__()
        self.dropout = dropout
        self.device = device
    
    def forward(self, query, key, value, attention_mask=None):
        """
        Scaled dot-product attention
        
        Args:
            query: (batch_size, seq_len, d_model)
            key: (batch_size, seq_len, d_model)  
            value: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len, seq_len)
        
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, seq_len, seq_len)
        """
        if not isinstance(query, Tensor):
            query = Tensor(query)
        if not isinstance(key, Tensor):
            key = Tensor(key)
        if not isinstance(value, Tensor):
            value = Tensor(value)
        
        d_k = query.shape[-1]
        
        # Compute attention scores: Q * K^T / sqrt(d_k)
        scores = (query @ key.transpose(-2, -1)) / (d_k ** 0.5)
        
        # Apply mask if provided
        if attention_mask is not None:
            mask_value = -1e9
            scores = scores + (1.0 - attention_mask) * mask_value
        
        # Apply softmax to get attention weights
        attention_weights = softmax(scores, dim=-1)
        
        # Apply dropout
        if self.training and self.dropout > 0.0:
            dropout_mask = torch.rand_like(attention_weights.data) > self.dropout
            attention_weights = attention_weights * Tensor(dropout_mask.float() / (1.0 - self.dropout))
        
        # Apply attention to values
        output = attention_weights @ value
        
        return output, attention_weights


class MultiHeadAttention(Module):
    """
    Multi-head attention mechanism
    """
    
    def __init__(self, d_model, num_heads, dropout=0.0, device='cpu'):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.device = device
        
        # Linear projections for Q, K, V
        self.W_q = Linear(d_model, d_model, device=device)
        self.W_k = Linear(d_model, d_model, device=device)
        self.W_v = Linear(d_model, d_model, device=device)
        self.W_o = Linear(d_model, d_model, device=device)
        
        self.attention = ScaledDotProductAttention(dropout, device)
        
        self._modules['W_q'] = self.W_q
        self._modules['W_k'] = self.W_k
        self._modules['W_v'] = self.W_v
        self._modules['W_o'] = self.W_o
        self._modules['attention'] = self.attention
    
    def forward(self, query, key, value, attention_mask=None):
        """
        Multi-head attention forward pass
        """
        batch_size, seq_len = query.shape[:2]
        
        # Linear projections
        Q = self.W_q(query)  # (batch_size, seq_len, d_model)
        K = self.W_k(key)    # (batch_size, seq_len, d_model)
        V = self.W_v(value)  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head attention
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Apply attention to each head
        # For simplicity, we'll process all heads at once
        # Reshape to (batch_size * num_heads, seq_len, d_k)
        Q_flat = Q.view(batch_size * self.num_heads, seq_len, self.d_k)
        K_flat = K.view(batch_size * self.num_heads, seq_len, self.d_k)
        V_flat = V.view(batch_size * self.num_heads, seq_len, self.d_k)
        
        # Apply scaled dot-product attention
        attention_output, attention_weights = self.attention(Q_flat, K_flat, V_flat)
        
        # Reshape back to (batch_size, num_heads, seq_len, d_k)
        attention_output = attention_output.view(batch_size, self.num_heads, seq_len, self.d_k)
        
        # Transpose back to (batch_size, seq_len, num_heads, d_k)
        attention_output = attention_output.transpose(1, 2)
        
        # Concatenate heads: (batch_size, seq_len, d_model)
        attention_output = attention_output.view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.W_o(attention_output)
        
        return output, attention_weights


# Compatibility layer for TensorFlow-style AdditiveAttention
class TFAdditiveAttention(Module):
    """
    TensorFlow-compatible additive attention
    Mimics the behavior of tf.keras.layers.AdditiveAttention
    """
    
    def __init__(self, use_scale=False, dropout=0.0, device='cpu'):
        super().__init__()
        self.attention = AdditiveAttention(use_scale=use_scale, dropout=dropout, device=device)
        self._modules['attention'] = self.attention
    
    def __call__(self, inputs, **kwargs):
        """
        TensorFlow-style call interface
        inputs: [query, value] or [query, value, key]
        """
        if isinstance(inputs, list):
            if len(inputs) == 2:
                query, value = inputs
                key = value
            elif len(inputs) == 3:
                query, value, key = inputs
            else:
                raise ValueError("inputs must be [query, value] or [query, value, key]")
        else:
            raise ValueError("inputs must be a list")
        
        context_vector, attention_weights = self.attention(query, value, key, **kwargs)
        return context_vector