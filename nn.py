"""
Core neural network components implemented from scratch
"""
from tensor import Tensor, zeros, ones, randn, xavier_uniform, xavier_normal, tanh, sigmoid, softmax
import torch
import math


class Parameter:
    """Parameter class for trainable parameters"""
    def __init__(self, data, requires_grad=True):
        if isinstance(data, Tensor):
            self.data = data
            self.data.requires_grad = requires_grad
        else:
            self.data = Tensor(data, requires_grad=requires_grad)
    
    def __repr__(self):
        return f"Parameter({self.data})"
    
    def zero_grad(self):
        self.data.zero_grad()
    
    def to(self, device):
        self.data = self.data.to(device)
        return self


class Module:
    """Base class for all neural network modules"""
    
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True
    
    def parameters(self):
        """Return all parameters"""
        params = []
        for param in self._parameters.values():
            params.append(param.data)
        for module in self._modules.values():
            params.extend(module.parameters())
        return params
    
    def named_parameters(self):
        """Return named parameters"""
        params = []
        for name, param in self._parameters.items():
            params.append((name, param.data))
        for module_name, module in self._modules.items():
            for name, param in module.named_parameters():
                params.append((f"{module_name}.{name}", param))
        return params
    
    def zero_grad(self):
        """Zero gradients of all parameters"""
        for param in self.parameters():
            param.zero_grad()
    
    def train(self, mode=True):
        """Set training mode"""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode"""
        return self.train(False)
    
    def to(self, device):
        """Move module to device"""
        for param in self._parameters.values():
            param.to(device)
        for module in self._modules.values():
            module.to(device)
        return self
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    """Linear layer: y = xW^T + b"""
    
    def __init__(self, in_features, out_features, bias=True, device='cpu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight matrix
        weight_data = randn(out_features, in_features, device=device)
        xavier_uniform(weight_data)
        self.weight = Parameter(weight_data)
        self._parameters['weight'] = self.weight
        
        if bias:
            bias_data = zeros(out_features, device=device)
            self.bias = Parameter(bias_data)
            self._parameters['bias'] = self.bias
        else:
            self.bias = None
    
    def forward(self, x):
        """Forward pass: y = xW^T + b"""
        # x: (batch_size, in_features) or (seq_len, batch_size, in_features)
        # weight: (out_features, in_features)
        # output: (batch_size, out_features) or (seq_len, batch_size, out_features)
        
        output = x @ self.weight.data.transpose(-1, -2)
        
        if self.bias is not None:
            output = output + self.bias.data
        
        return output


class Embedding(Module):
    """Embedding layer"""
    
    def __init__(self, num_embeddings, embedding_dim, device='cpu'):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize embedding matrix
        weight_data = randn(num_embeddings, embedding_dim, device=device)
        xavier_normal(weight_data)
        self.weight = Parameter(weight_data)
        self._parameters['weight'] = self.weight
    
    def forward(self, input_ids):
        """
        Forward pass
        input_ids: (batch_size, seq_len) - indices of tokens
        output: (batch_size, seq_len, embedding_dim)
        """
        if not isinstance(input_ids, Tensor):
            input_ids = Tensor(input_ids)
        
        # Convert indices to integer for indexing
        indices = input_ids.data.long()
        
        # Index into embedding matrix
        embeddings = self.weight.data.data[indices]  # Get raw torch tensor for indexing
        result = Tensor(embeddings, requires_grad=self.weight.data.requires_grad)
        
        def backward_fn(grad):
            if self.weight.data.requires_grad:
                # Accumulate gradients for each embedding
                grad_weight = torch.zeros_like(self.weight.data.data)
                grad_weight.index_add_(0, indices.flatten(), grad.data.view(-1, self.embedding_dim))
                self.weight.data.backward(grad_weight)
        
        if result.requires_grad:
            result._backward_fn = backward_fn
        
        return result


class LayerNorm(Module):
    """Layer normalization"""
    
    def __init__(self, normalized_shape, eps=1e-5, device='cpu'):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.weight = Parameter(ones(*normalized_shape, device=device))
        self.bias = Parameter(zeros(*normalized_shape, device=device))
        self._parameters['weight'] = self.weight
        self._parameters['bias'] = self.bias
    
    def forward(self, x):
        """
        Forward pass
        x: (..., *normalized_shape)
        """
        # Calculate mean and variance over the last len(normalized_shape) dimensions
        dims = list(range(-len(self.normalized_shape), 0))
        
        mean = x.mean(dim=dims, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
        
        # Normalize
        x_norm = (x - mean) / ((var + self.eps) ** 0.5)
        
        # Scale and shift
        output = self.weight.data * x_norm + self.bias.data
        
        return output


class Dropout(Module):
    """Dropout layer"""
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        # Generate random mask
        mask = torch.rand_like(x.data) > self.p
        output = x * Tensor(mask.float() / (1.0 - self.p))
        
        return output


class Tanh(Module):
    """Tanh activation function"""
    
    def forward(self, x):
        return tanh(x)


class Sigmoid(Module):
    """Sigmoid activation function"""
    
    def forward(self, x):
        return sigmoid(x)


class Softmax(Module):
    """Softmax activation function"""
    
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return softmax(x, dim=self.dim)


class ReLU(Module):
    """ReLU activation function"""
    
    def forward(self, x):
        result_data = torch.relu(x.data)
        result = Tensor(result_data, requires_grad=x.requires_grad)
        
        def backward_fn(grad):
            if x.requires_grad:
                grad_input = grad * (x.data > 0).float()
                x.backward(grad_input)
        
        if result.requires_grad:
            result._backward_fn = backward_fn
        
        return result


# Sequence utilities
class Sequential(Module):
    """Sequential container of modules"""
    
    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            self._modules[str(i)] = module
    
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


# Loss functions
def cross_entropy_loss(predictions, targets):
    """
    Cross entropy loss
    predictions: (batch_size, num_classes) - logits
    targets: (batch_size,) - class indices
    """
    if not isinstance(predictions, Tensor):
        predictions = Tensor(predictions)
    if not isinstance(targets, Tensor):
        targets = Tensor(targets)
    
    # Apply softmax to get probabilities
    probs = softmax(predictions, dim=-1)
    
    # Convert targets to long for indexing
    targets_long = targets.data.long()
    
    # Get the probability of the correct class for each sample
    batch_size = predictions.shape[0]
    correct_probs = probs.data[torch.arange(batch_size), targets_long]
    
    # Compute negative log likelihood
    loss = -correct_probs.log().mean()
    
    return loss


def sparse_categorical_crossentropy(predictions, targets):
    """
    Sparse categorical crossentropy loss
    predictions: (batch_size, num_classes) or (seq_len, batch_size, num_classes) - logits  
    targets: (batch_size,) or (seq_len, batch_size) - class indices
    """
    if not isinstance(predictions, Tensor):
        predictions = Tensor(predictions)
    if not isinstance(targets, Tensor):
        targets = Tensor(targets)
    
    original_shape = predictions.shape
    
    # Reshape for processing
    if len(original_shape) == 3:  # sequence case
        seq_len, batch_size, num_classes = original_shape
        predictions_flat = predictions.view(seq_len * batch_size, num_classes)
        targets_flat = targets.view(-1)
    else:  # batch case
        predictions_flat = predictions
        targets_flat = targets
    
    # Apply softmax to predictions
    log_probs = predictions_flat - predictions_flat.max(dim=-1, keepdim=True).detach()  
    log_probs = log_probs - log_probs.exp().sum(dim=-1, keepdim=True).log()
    
    # Get negative log likelihood
    batch_size = predictions_flat.shape[0]
    targets_long = targets_flat.data.long()
    
    # Create result tensor
    loss_data = -log_probs.data[torch.arange(batch_size), targets_long].mean()
    loss = Tensor(loss_data, requires_grad=predictions.requires_grad)
    
    def backward_fn(grad):
        if predictions.requires_grad:
            # Softmax gradient
            probs = log_probs.exp()
            grad_input = probs.data.clone()
            grad_input[torch.arange(batch_size), targets_long] -= 1.0
            grad_input = grad_input * grad.data / batch_size
            
            # Reshape back to original shape if needed
            if len(original_shape) == 3:
                grad_input = grad_input.view(original_shape)
            
            predictions.backward(grad_input)
    
    if loss.requires_grad:
        loss._backward_fn = backward_fn
    
    return loss