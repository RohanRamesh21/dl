"""
Optimizers and loss functions implemented from scratch
"""
from tensor import Tensor, zeros, ones
import torch
import math


class Optimizer:
    """Base optimizer class"""
    
    def __init__(self, parameters):
        self.param_groups = [{'params': parameters}]
    
    def zero_grad(self):
        """Zero gradients of all parameters"""
        for group in self.param_groups:
            for param in group['params']:
                param.zero_grad()
    
    def step(self):
        """Perform optimization step"""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Initialize momentum buffers
        for group in self.param_groups:
            for param in group['params']:
                if not hasattr(param, 'momentum_buffer'):
                    param.momentum_buffer = zeros(*param.shape, device=param.device)
    
    def step(self):
        """Perform SGD step"""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # Add weight decay
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param.data
                
                # Apply momentum
                if self.momentum != 0:
                    param.momentum_buffer = self.momentum * param.momentum_buffer + grad
                    grad = param.momentum_buffer
                
                # Update parameters
                with torch.no_grad():
                    param.data.data -= self.lr * grad.data


class Adam(Optimizer):
    """Adam optimizer implemented from scratch"""
    
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        
        # Initialize moment estimates
        for group in self.param_groups:
            for param in group['params']:
                if not hasattr(param, 'm'):
                    param.m = zeros(*param.shape, device=param.device)  # First moment
                if not hasattr(param, 'v'):
                    param.v = zeros(*param.shape, device=param.device)  # Second moment
    
    def step(self):
        """Perform Adam optimization step"""
        self.step_count += 1
        
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # Add weight decay
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param.data
                
                # Update biased first moment estimate
                param.m = self.beta1 * param.m + (1 - self.beta1) * grad
                
                # Update biased second raw moment estimate
                param.v = self.beta2 * param.v + (1 - self.beta2) * (grad ** 2)
                
                # Compute bias-corrected first moment estimate
                m_hat = param.m / (1 - self.beta1 ** self.step_count)
                
                # Compute bias-corrected second raw moment estimate
                v_hat = param.v / (1 - self.beta2 ** self.step_count)
                
                # Update parameters
                with torch.no_grad():
                    param.data.data -= self.lr * m_hat.data / (torch.sqrt(v_hat.data) + self.eps)


class AdamW(Optimizer):
    """AdamW optimizer (Adam with decoupled weight decay)"""
    
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        
        # Initialize moment estimates
        for group in self.param_groups:
            for param in group['params']:
                if not hasattr(param, 'm'):
                    param.m = zeros(*param.shape, device=param.device)
                if not hasattr(param, 'v'):
                    param.v = zeros(*param.shape, device=param.device)
    
    def step(self):
        """Perform AdamW optimization step"""
        self.step_count += 1
        
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # Update biased first moment estimate
                param.m = self.beta1 * param.m + (1 - self.beta1) * grad
                
                # Update biased second raw moment estimate
                param.v = self.beta2 * param.v + (1 - self.beta2) * (grad ** 2)
                
                # Compute bias-corrected first moment estimate
                m_hat = param.m / (1 - self.beta1 ** self.step_count)
                
                # Compute bias-corrected second raw moment estimate
                v_hat = param.v / (1 - self.beta2 ** self.step_count)
                
                # Update parameters with decoupled weight decay
                with torch.no_grad():
                    # Adam update
                    param.data.data -= self.lr * m_hat.data / (torch.sqrt(v_hat.data) + self.eps)
                    # Weight decay
                    param.data.data -= self.lr * self.weight_decay * param.data.data


class RMSprop(Optimizer):
    """RMSprop optimizer"""
    
    def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        
        # Initialize moving average and momentum
        for group in self.param_groups:
            for param in group['params']:
                if not hasattr(param, 'square_avg'):
                    param.square_avg = zeros(*param.shape, device=param.device)
                if self.momentum > 0 and not hasattr(param, 'momentum_buffer'):
                    param.momentum_buffer = zeros(*param.shape, device=param.device)
    
    def step(self):
        """Perform RMSprop step"""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # Add weight decay
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param.data
                
                # Update moving average of squared gradients
                param.square_avg = self.alpha * param.square_avg + (1 - self.alpha) * (grad ** 2)
                
                # Compute update
                avg = param.square_avg
                update = grad / (avg ** 0.5 + self.eps)
                
                # Apply momentum if specified
                if self.momentum > 0:
                    param.momentum_buffer = self.momentum * param.momentum_buffer + update
                    update = param.momentum_buffer
                
                # Update parameters
                with torch.no_grad():
                    param.data.data -= self.lr * update.data


# Learning rate schedulers
class LRScheduler:
    """Base learning rate scheduler"""
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_count = 0
    
    def step(self, epoch=None):
        """Update learning rate"""
        raise NotImplementedError
    
    def get_lr(self):
        """Get current learning rate"""
        raise NotImplementedError


class StepLR(LRScheduler):
    """Step learning rate scheduler"""
    
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = optimizer.lr
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.step_count
        
        self.step_count += 1
        new_lr = self.base_lr * (self.gamma ** (epoch // self.step_size))
        self.optimizer.lr = new_lr
    
    def get_lr(self):
        return self.optimizer.lr


class ExponentialLR(LRScheduler):
    """Exponential learning rate scheduler"""
    
    def __init__(self, optimizer, gamma):
        super().__init__(optimizer)
        self.gamma = gamma
        self.base_lr = optimizer.lr
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.step_count
        
        self.step_count += 1
        new_lr = self.base_lr * (self.gamma ** epoch)
        self.optimizer.lr = new_lr
    
    def get_lr(self):
        return self.optimizer.lr


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler"""
    
    def __init__(self, optimizer, T_max, eta_min=0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.step_count
        
        self.step_count += 1
        new_lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        self.optimizer.lr = new_lr
    
    def get_lr(self):
        return self.optimizer.lr


# Loss Functions
class Loss:
    """Base loss function"""
    
    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)
    
    def forward(self, predictions, targets):
        raise NotImplementedError


class MSELoss(Loss):
    """Mean Squared Error loss"""
    
    def forward(self, predictions, targets):
        if not isinstance(predictions, Tensor):
            predictions = Tensor(predictions)
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)
        
        diff = predictions - targets
        loss = (diff ** 2).mean()
        return loss


class MAELoss(Loss):
    """Mean Absolute Error loss"""
    
    def forward(self, predictions, targets):
        if not isinstance(predictions, Tensor):
            predictions = Tensor(predictions)
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)
        
        diff = predictions - targets
        # Implement abs as sqrt(x^2)
        abs_diff = (diff ** 2) ** 0.5
        loss = abs_diff.mean()
        return loss


class CrossEntropyLoss(Loss):
    """Cross entropy loss"""
    
    def forward(self, predictions, targets):
        return cross_entropy_loss(predictions, targets)


class SparseCategoricalCrossentropy(Loss):
    """Sparse categorical crossentropy loss (matches TensorFlow's implementation)"""
    
    def __init__(self, from_logits=True):
        self.from_logits = from_logits
    
    def forward(self, predictions, targets):
        """
        Compute sparse categorical crossentropy loss
        
        Args:
            predictions: Logits tensor (batch_size, num_classes) or (seq_len, batch_size, num_classes)
            targets: Target indices (batch_size,) or (seq_len, batch_size)
        
        Returns:
            loss: Scalar loss value
        """
        if not isinstance(predictions, Tensor):
            predictions = Tensor(predictions, requires_grad=True)
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)
        
        original_shape = predictions.shape
        
        # Handle sequence case (seq_len, batch_size, num_classes)
        if len(original_shape) == 3:
            seq_len, batch_size, num_classes = original_shape
            predictions_flat = predictions.view(seq_len * batch_size, num_classes)
            targets_flat = targets.view(-1)
        else:
            predictions_flat = predictions
            targets_flat = targets
        
        # Convert to probabilities if from_logits=True
        if self.from_logits:
            # Numerical stability: subtract max
            max_vals = predictions_flat.max(dim=-1, keepdim=True)
            if isinstance(max_vals, tuple):
                max_vals = max_vals[0] if hasattr(max_vals, '__getitem__') else max_vals
            shifted_logits = predictions_flat - max_vals
            
            # Compute log probabilities  
            exp_logits = shifted_logits.exp()
            log_sum_exp = (exp_logits.sum(dim=-1, keepdim=True)).log()
            log_probs = shifted_logits - log_sum_exp
        else:
            # Already probabilities, just take log
            log_probs = predictions_flat.log()
        
        # Get the log probability of the correct class
        batch_size = predictions_flat.shape[0]
        targets_long = targets_flat.data.long()
        
        # Index into log_probs to get correct class probabilities
        correct_log_probs = log_probs.data[torch.arange(batch_size), targets_long]
        
        # Compute negative log likelihood
        loss = -Tensor(correct_log_probs.mean(), requires_grad=predictions.requires_grad)
        
        def backward_fn(grad):
            if predictions.requires_grad:
                # Softmax derivative
                if self.from_logits:
                    probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
                else:
                    probs = predictions_flat
                
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


# Import the existing function for compatibility
from nn import sparse_categorical_crossentropy, cross_entropy_loss