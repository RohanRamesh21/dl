"""
Custom tensor implementation with automatic differentiation
Using PyTorch tensors for data storage and CUDA support, but implementing autograd from scratch
"""
import torch
import math
from typing import Optional, Tuple, List, Union, Callable


class Tensor:
    """Custom tensor class with automatic differentiation"""
    
    def __init__(self, data, requires_grad=False, device='cpu'):
        if isinstance(data, (int, float)):
            self.data = torch.tensor(float(data), device=device, dtype=torch.float32)
        elif isinstance(data, (list, tuple)):
            self.data = torch.tensor(data, device=device, dtype=torch.float32)
        elif isinstance(data, torch.Tensor):
            self.data = data.to(device).float()
        else:
            self.data = torch.tensor(data, device=device, dtype=torch.float32)
            
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self._backward_fn = None
        
        if requires_grad:
            self.grad = torch.zeros_like(self.data)
    
    @property
    def shape(self):
        return self.data.shape
    
    @property 
    def device(self):
        return self.data.device
    
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    def zero_grad(self):
        """Zero out gradients"""
        if self.grad is not None:
            self.grad.zero_()
    
    def backward(self, gradient=None):
        """Compute gradients via backpropagation"""
        if not self.requires_grad:
            return
            
        if gradient is None:
            if self.data.numel() != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            gradient = torch.ones_like(self.data)
        
        # Prevent infinite recursion by checking if we're already processing this tensor
        if hasattr(self, '_processing_backward') and self._processing_backward:
            return
        
        self._processing_backward = True
        
        try:
            if self.grad is None:
                self.grad = torch.zeros_like(self.data)
                
            self.grad += gradient
            
            if self._backward_fn is not None:
                self._backward_fn(gradient)
        finally:
            self._processing_backward = False
    
    def detach(self):
        """Return a new tensor detached from computation graph"""
        result = Tensor(self.data.clone(), requires_grad=False)
        return result
    
    # Math operations
    def __add__(self, other):
        return add(self, other)
    
    def __radd__(self, other):
        return add(other, self)
    
    def __sub__(self, other):
        return sub(self, other)
    
    def __rsub__(self, other):
        return sub(other, self)
    
    def __mul__(self, other):
        return mul(self, other)
    
    def __rmul__(self, other):
        return mul(other, self)
    
    def __truediv__(self, other):
        return div(self, other)
    
    def __rtruediv__(self, other):
        return div(other, self)
    
    def __matmul__(self, other):
        return matmul(self, other)
    
    def __pow__(self, other):
        return pow(self, other)
    
    def __neg__(self):
        return mul(self, -1)
    
    def sum(self, dim=None, keepdim=False):
        return tensor_sum(self, dim, keepdim)
    
    def mean(self, dim=None, keepdim=False):
        return tensor_mean(self, dim, keepdim)
    
    def exp(self):
        return exp(self)
    
    def log(self):
        return log(self)
    
    def max(self, dim=None, keepdim=False):
        if dim is None:
            result_data = self.data.max()
            result = Tensor(result_data, requires_grad=self.requires_grad)
            
            def backward_fn(grad):
                if self.requires_grad:
                    # Gradient flows to the maximum element
                    max_mask = (self.data == result_data).float()
                    grad_input = grad * max_mask / max_mask.sum()
                    self.backward(grad_input)
            
            if result.requires_grad:
                result._backward_fn = backward_fn
                
            return result
        else:
            max_result = self.data.max(dim=dim, keepdim=keepdim)
            values = Tensor(max_result.values, requires_grad=self.requires_grad)
            # For simplicity, return just values for now
            return values
    
    def view(self, *shape):
        return view(self, shape)
    
    def reshape(self, *shape):
        return view(self, shape)
    
    def transpose(self, dim0, dim1):
        return transpose(self, dim0, dim1)
    
    def permute(self, *dims):
        return permute(self, dims)
    
    def unsqueeze(self, dim):
        return unsqueeze(self, dim)
    
    def squeeze(self, dim=None):
        return squeeze(self, dim)
    
    def to(self, device):
        """Move tensor to device"""
        result = Tensor(self.data.to(device), requires_grad=self.requires_grad)
        if self.grad is not None:
            result.grad = self.grad.to(device)
        return result


def _ensure_tensor(x):
    """Convert input to Tensor if it isn't already"""
    if not isinstance(x, Tensor):
        return Tensor(x)
    return x


# Arithmetic operations
def add(a, b):
    """Element-wise addition"""
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    
    result_data = a.data + b.data
    result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
    def backward_fn(grad):
        if a.requires_grad:
            # Handle broadcasting
            grad_a = grad
            # Sum out added dims and keepdim=True for broadcasted dims
            ndims_added = len(grad.shape) - len(a.data.shape)
            for i in range(ndims_added):
                grad_a = grad_a.sum(dim=0)
            
            for i, (dim_a, dim_grad) in enumerate(zip(a.data.shape, grad_a.shape)):
                if dim_a == 1 and dim_grad > 1:
                    grad_a = grad_a.sum(dim=i, keepdim=True)
            
            a.backward(grad_a)
        
        if b.requires_grad:
            grad_b = grad
            # Handle broadcasting for b
            ndims_added = len(grad.shape) - len(b.data.shape)
            for i in range(ndims_added):
                grad_b = grad_b.sum(dim=0)
            
            for i, (dim_b, dim_grad) in enumerate(zip(b.data.shape, grad_b.shape)):
                if dim_b == 1 and dim_grad > 1:
                    grad_b = grad_b.sum(dim=i, keepdim=True)
            
            b.backward(grad_b)
    
    if result.requires_grad:
        result._backward_fn = backward_fn
    
    return result


def sub(a, b):
    """Element-wise subtraction"""
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    
    result_data = a.data - b.data
    result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
    def backward_fn(grad):
        if a.requires_grad:
            grad_a = grad
            ndims_added = len(grad.shape) - len(a.data.shape)
            for i in range(ndims_added):
                grad_a = grad_a.sum(dim=0)
            
            for i, (dim_a, dim_grad) in enumerate(zip(a.data.shape, grad_a.shape)):
                if dim_a == 1 and dim_grad > 1:
                    grad_a = grad_a.sum(dim=i, keepdim=True)
            
            a.backward(grad_a)
        
        if b.requires_grad:
            grad_b = -grad
            ndims_added = len(grad.shape) - len(b.data.shape)
            for i in range(ndims_added):
                grad_b = grad_b.sum(dim=0)
            
            for i, (dim_b, dim_grad) in enumerate(zip(b.data.shape, grad_b.shape)):
                if dim_b == 1 and dim_grad > 1:
                    grad_b = grad_b.sum(dim=i, keepdim=True)
            
            b.backward(grad_b)
    
    if result.requires_grad:
        result._backward_fn = backward_fn
    
    return result


def mul(a, b):
    """Element-wise multiplication"""
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    
    result_data = a.data * b.data
    result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
    def backward_fn(grad):
        if a.requires_grad:
            grad_a = grad * b.data
            ndims_added = len(grad.shape) - len(a.data.shape)
            for i in range(ndims_added):
                grad_a = grad_a.sum(dim=0)
            
            for i, (dim_a, dim_grad) in enumerate(zip(a.data.shape, grad_a.shape)):
                if dim_a == 1 and dim_grad > 1:
                    grad_a = grad_a.sum(dim=i, keepdim=True)
            
            a.backward(grad_a)
        
        if b.requires_grad:
            grad_b = grad * a.data
            ndims_added = len(grad.shape) - len(b.data.shape)
            for i in range(ndims_added):
                grad_b = grad_b.sum(dim=0)
            
            for i, (dim_b, dim_grad) in enumerate(zip(b.data.shape, grad_b.shape)):
                if dim_b == 1 and dim_grad > 1:
                    grad_b = grad_b.sum(dim=i, keepdim=True)
            
            b.backward(grad_b)
    
    if result.requires_grad:
        result._backward_fn = backward_fn
    
    return result


def div(a, b):
    """Element-wise division"""
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    
    result_data = a.data / b.data
    result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
    def backward_fn(grad):
        if a.requires_grad:
            grad_a = grad / b.data
            ndims_added = len(grad.shape) - len(a.data.shape)
            for i in range(ndims_added):
                grad_a = grad_a.sum(dim=0)
            
            for i, (dim_a, dim_grad) in enumerate(zip(a.data.shape, grad_a.shape)):
                if dim_a == 1 and dim_grad > 1:
                    grad_a = grad_a.sum(dim=i, keepdim=True)
            
            a.backward(grad_a)
        
        if b.requires_grad:
            grad_b = -grad * a.data / (b.data ** 2)
            ndims_added = len(grad.shape) - len(b.data.shape)
            for i in range(ndims_added):
                grad_b = grad_b.sum(dim=0)
            
            for i, (dim_b, dim_grad) in enumerate(zip(b.data.shape, grad_b.shape)):
                if dim_b == 1 and dim_grad > 1:
                    grad_b = grad_b.sum(dim=i, keepdim=True)
            
            b.backward(grad_b)
    
    if result.requires_grad:
        result._backward_fn = backward_fn
    
    return result


def pow(a, b):
    """Element-wise power"""
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    
    result_data = torch.pow(a.data, b.data)
    result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
    def backward_fn(grad):
        if a.requires_grad:
            grad_a = grad * b.data * torch.pow(a.data, b.data - 1)
            a.backward(grad_a)
        
        if b.requires_grad:
            grad_b = grad * torch.pow(a.data, b.data) * torch.log(a.data)
            b.backward(grad_b)
    
    if result.requires_grad:
        result._backward_fn = backward_fn
    
    return result


def matmul(a, b):
    """Matrix multiplication"""
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    
    result_data = torch.matmul(a.data, b.data)
    result = Tensor(result_data, requires_grad=(a.requires_grad or b.requires_grad))
    
    def backward_fn(grad):
        if a.requires_grad:
            if len(b.data.shape) == 1:
                grad_a = torch.outer(grad, b.data)
            elif len(a.data.shape) == 1:
                grad_a = torch.matmul(grad, b.data.transpose(-1, -2))
            else:
                grad_a = torch.matmul(grad, b.data.transpose(-1, -2))
            a.backward(grad_a)
        
        if b.requires_grad:
            if len(a.data.shape) == 1:
                grad_b = torch.outer(a.data, grad)
            elif len(b.data.shape) == 1:
                grad_b = torch.matmul(a.data.transpose(-1, -2), grad)
            else:
                grad_b = torch.matmul(a.data.transpose(-1, -2), grad)
            b.backward(grad_b)
    
    if result.requires_grad:
        result._backward_fn = backward_fn
    
    return result


# Shape operations
def view(tensor, shape):
    """Reshape tensor"""
    result_data = tensor.data.view(*shape)
    result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
    def backward_fn(grad):
        if tensor.requires_grad:
            tensor.backward(grad.view(tensor.data.shape))
    
    if result.requires_grad:
        result._backward_fn = backward_fn
    
    return result


def transpose(tensor, dim0, dim1):
    """Transpose two dimensions"""
    result_data = tensor.data.transpose(dim0, dim1)
    result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
    def backward_fn(grad):
        if tensor.requires_grad:
            tensor.backward(grad.transpose(dim0, dim1))
    
    if result.requires_grad:
        result._backward_fn = backward_fn
    
    return result


def permute(tensor, dims):
    """Permute dimensions"""
    result_data = tensor.data.permute(*dims)
    result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
    def backward_fn(grad):
        if tensor.requires_grad:
            # Invert permutation
            inv_perm = [0] * len(dims)
            for i, d in enumerate(dims):
                inv_perm[d] = i
            tensor.backward(grad.permute(*inv_perm))
    
    if result.requires_grad:
        result._backward_fn = backward_fn
    
    return result


def unsqueeze(tensor, dim):
    """Add dimension"""
    result_data = tensor.data.unsqueeze(dim)
    result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
    def backward_fn(grad):
        if tensor.requires_grad:
            tensor.backward(grad.squeeze(dim))
    
    if result.requires_grad:
        result._backward_fn = backward_fn
    
    return result


def squeeze(tensor, dim=None):
    """Remove dimension"""
    if dim is None:
        result_data = tensor.data.squeeze()
    else:
        result_data = tensor.data.squeeze(dim)
    result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
    def backward_fn(grad):
        if tensor.requires_grad:
            if dim is None:
                tensor.backward(grad.view(tensor.data.shape))
            else:
                tensor.backward(grad.unsqueeze(dim))
    
    if result.requires_grad:
        result._backward_fn = backward_fn
    
    return result


# Reduction operations
def tensor_sum(tensor, dim=None, keepdim=False):
    """Sum tensor elements"""
    result_data = tensor.data.sum(dim=dim, keepdim=keepdim)
    result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
    def backward_fn(grad):
        if tensor.requires_grad:
            if dim is None:
                grad_expanded = grad.expand(tensor.data.shape)
            else:
                if not keepdim:
                    grad = grad.unsqueeze(dim)
                grad_expanded = grad.expand(tensor.data.shape)
            tensor.backward(grad_expanded)
    
    if result.requires_grad:
        result._backward_fn = backward_fn
    
    return result


def tensor_mean(tensor, dim=None, keepdim=False):
    """Mean of tensor elements"""
    result_data = tensor.data.mean(dim=dim, keepdim=keepdim)
    result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
    def backward_fn(grad):
        if tensor.requires_grad:
            if dim is None:
                grad_expanded = grad.expand(tensor.data.shape) / tensor.data.numel()
            else:
                size = tensor.data.shape[dim]
                if not keepdim:
                    grad = grad.unsqueeze(dim)
                grad_expanded = grad.expand(tensor.data.shape) / size
            tensor.backward(grad_expanded)
    
    if result.requires_grad:
        result._backward_fn = backward_fn
    
    return result


# Activation functions
def tanh(tensor):
    """Hyperbolic tangent activation"""
    result_data = torch.tanh(tensor.data)
    result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
    def backward_fn(grad):
        if tensor.requires_grad:
            grad_input = grad * (1 - result_data ** 2)
            tensor.backward(grad_input)
    
    if result.requires_grad:
        result._backward_fn = backward_fn
    
    return result


def sigmoid(tensor):
    """Sigmoid activation"""
    result_data = torch.sigmoid(tensor.data)
    result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
    def backward_fn(grad):
        if tensor.requires_grad:
            grad_input = grad * result_data * (1 - result_data)
            tensor.backward(grad_input)
    
    if result.requires_grad:
        result._backward_fn = backward_fn
    
    return result


def softmax(tensor, dim=-1):
    """Softmax activation"""
    # Numerical stability: subtract max
    max_vals = tensor.data.max(dim=dim, keepdim=True).values
    shifted = tensor.data - max_vals
    exp_vals = torch.exp(shifted)
    sum_exp = exp_vals.sum(dim=dim, keepdim=True)
    result_data = exp_vals / sum_exp
    
    result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
    def backward_fn(grad):
        if tensor.requires_grad:
            # Jacobian of softmax: S_ij = s_i(δ_ij - s_j)
            s = result_data
            grad_input = s * (grad - (grad * s).sum(dim=dim, keepdim=True))
            tensor.backward(grad_input)
    
    if result.requires_grad:
        result._backward_fn = backward_fn
    
    return result


def log(tensor):
    """Natural logarithm"""
    result_data = torch.log(tensor.data)
    result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
    def backward_fn(grad):
        if tensor.requires_grad:
            grad_input = grad / tensor.data
            tensor.backward(grad_input)
    
    if result.requires_grad:
        result._backward_fn = backward_fn
    
    return result


def exp(tensor):
    """Exponential function"""
    result_data = torch.exp(tensor.data)
    result = Tensor(result_data, requires_grad=tensor.requires_grad)
    
    def backward_fn(grad):
        if tensor.requires_grad:
            grad_input = grad * result_data
            tensor.backward(grad_input)
    
    if result.requires_grad:
        result._backward_fn = backward_fn
    
    return result


# Utility functions
def zeros(*shape, requires_grad=False, device='cpu'):
    """Create tensor of zeros"""
    return Tensor(torch.zeros(*shape, device=device), requires_grad=requires_grad)


def ones(*shape, requires_grad=False, device='cpu'):
    """Create tensor of ones"""
    return Tensor(torch.ones(*shape, device=device), requires_grad=requires_grad)


def randn(*shape, requires_grad=False, device='cpu'):
    """Create tensor with random normal values"""
    return Tensor(torch.randn(*shape, device=device), requires_grad=requires_grad)


def rand(*shape, requires_grad=False, device='cpu'):
    """Create tensor with random uniform values"""
    return Tensor(torch.rand(*shape, device=device), requires_grad=requires_grad)


def xavier_uniform(tensor):
    """Xavier uniform initialization"""
    fan_in = tensor.shape[-2] if len(tensor.shape) > 1 else tensor.shape[-1]
    fan_out = tensor.shape[-1]
    std = math.sqrt(2.0 / (fan_in + fan_out))
    with torch.no_grad():
        tensor.data.uniform_(-std, std)


def xavier_normal(tensor):
    """Xavier normal initialization"""
    fan_in = tensor.shape[-2] if len(tensor.shape) > 1 else tensor.shape[-1]
    fan_out = tensor.shape[-1]
    std = math.sqrt(2.0 / (fan_in + fan_out))
    with torch.no_grad():
        tensor.data.normal_(0, std)