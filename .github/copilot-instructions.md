# Instructions for GitHub Copilot

## Problem 1: Inconsistent Teacher Forcing in Validation

### Problem Explanation:
**Current Issue**: During validation, we're using 100% teacher forcing (`model(encoder_inputs, decoder_inputs)`) while training uses variable teacher forcing ratios. This creates an unfair comparison because:
- Training sees ground truth tokens (teacher forcing)
- Validation also sees ground truth tokens (always teacher forcing)
- But during inference (real translation), the model must use its own predictions

**Impact**: Validation metrics become overly optimistic and don't reflect real translation performance.

### Solution Instructions:
```python
# REPLACE the current validation logic in train_model_enhanced():

# CURRENT (problematic):
predictions = model(enc_batch, dec_input_batch)  # Always 100% teacher forcing

# WITH THIS (consistent):
# Use the same teacher forcing strategy as training during validation
if teacher_forcing_ratio > 0:  # Match training's teacher forcing ratio
    target_sequence = torch.cat([dec_input_batch, dec_target_batch[:, -1:]], dim=1)
    predictions = model.forward_with_teacher_forcing(
        enc_batch, target_sequence, teacher_forcing_ratio, 
        sos_id=data_dict['sos_id'], eos_id=data_dict['eos_id']
    )
else:
    # For autoregressive validation (no teacher forcing)
    predictions = autoregressive_decode(
        model, enc_batch, max_length=dec_target_batch.size(1),
        sos_id=data_dict['sos_id'], eos_id=data_dict['eos_id']
    )
```

## Problem 2: State Projection Logic Issues

### Problem Explanation:
**Current Issue**: The state projection logic in `Decoder.__call__()` has fragile type checking that breaks with multi-layer bidirectional encoders:
```python
if isinstance(initial_state[0], torch.Tensor) and initial_state[0].dim() == 2:
    # Single layer case
```
This doesn't handle:
- Multi-layer states (stacked tensors)
- Bidirectional encoder states (different dimensions)
- Mixed state representations

### Solution Instructions:
```python
# REPLACE the state projection logic in Decoder.__call__():

# CURRENT (fragile):
if self.h_projection is not None:
    if isinstance(initial_state[0], torch.Tensor) and initial_state[0].dim() == 2:
        # Single layer case
        h_init = self.h_projection(initial_state[0])
        c_init = self.c_projection(initial_state[1])
        projected_state = (h_init, c_init)
    else:
        # Multi-layer case - problematic logic
        h_layers = [self.h_projection(h) for h in initial_state[0]]
        c_layers = [self.c_projection(c) for c in initial_state[1]]

# WITH THIS (robust):
def _project_encoder_states(self, encoder_states):
    """Unified state projection handling all encoder configurations"""
    state_h, state_c = encoder_states
    
    if self.h_projection is None:
        return encoder_states
    
    # Handle all cases: single-layer, multi-layer, bidirectional
    if isinstance(state_h, torch.Tensor):
        if state_h.dim() == 2:
            # Single layer: (batch, hidden*num_directions)
            return (self.h_projection(state_h), self.c_projection(state_c))
        elif state_h.dim() == 3:
            # Multi-layer: (num_layers, batch, hidden*num_directions)
            return (self.h_projection(state_h), self.c_projection(state_c))
    elif isinstance(state_h, (list, tuple)):
        # List of layer states
        h_proj = [self.h_projection(h) for h in state_h]
        c_proj = [self.c_projection(c) for c in state_c]
        return (h_proj, c_proj)
    
    return encoder_states  # Fallback
```

## Problem 3: Bidirectional LSTM State Concatenation

### Problem Explanation:
**Current Issue**: In `LSTM._bidirectional_forward()`, the state concatenation is incorrect for multi-layer cases:
```python
final_h = [torch.cat([h_f, h_b], dim=-1) for h_f, h_b in zip(h_forward, h_backward)]
```
This creates a list of concatenated states, but the LSTM interface expects properly stacked tensors for multi-layer configurations.

### Solution Instructions:
```python
# REPLACE the bidirectional state concatenation in LSTM._bidirectional_forward():

# CURRENT (incorrect stacking):
final_h = [torch.cat([h_f, h_b], dim=-1) for h_f, h_b in zip(h_forward, h_backward)]
final_c = [torch.cat([c_f, c_b], dim=-1) for c_f, c_b in zip(c_forward, c_backward)]

if self.num_layers == 1:
    return output, (final_h[0], final_c[0])
else:
    return output, (torch.stack(final_h), torch.stack(final_c))

# WITH THIS (correct stacking):
# First stack layers, then concatenate bidirectional dimensions
if self.num_layers == 1:
    final_h = torch.cat([h_forward[0], h_backward[0]], dim=-1)
    final_c = torch.cat([c_forward[0], c_backward[0]], dim=-1)
else:
    # Stack layers first: (num_layers, batch, hidden_size)
    h_forward_stacked = torch.stack(h_forward)
    h_backward_stacked = torch.stack(h_backward)
    c_forward_stacked = torch.stack(c_forward)
    c_backward_stacked = torch.stack(c_backward)
    
    # Concatenate bidirectional dimensions: (num_layers, batch, hidden_size*2)
    final_h = torch.cat([h_forward_stacked, h_backward_stacked], dim=-1)
    final_c = torch.cat([c_forward_stacked, c_backward_stacked], dim=-1)

return output, (final_h, final_c)
```

## Problem 4: Gradient Clipping Implementation

### Problem Explanation:
**Current Issue**: The manual gradient clipping implementation is:
- **Inefficient**: Loops through all parameters twice
- **Potentially incorrect**: Manual norm calculation and scaling can have numerical issues
- **Hard to maintain**: Custom implementation instead of using battle-tested PyTorch functions

### Solution Instructions:
```python
# REPLACE the manual gradient clipping in train_step():

# CURRENT (manual implementation):
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

# WITH THIS (using PyTorch built-in):
if clip_grad_norm > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
```

## Implementation Priority:

1. **Fix Gradient Clipping First** - Most critical for training stability
2. **Fix Bidirectional State Concatenation** - Affects model correctness  
3. **Fix State Projection Logic** - Improves robustness
4. **Fix Teacher Forcing Consistency** - Improves evaluation accuracy

## Testing Instructions:
After implementing fixes, add these validation checks:

```python
# Test gradient clipping
assert hasattr(torch.nn.utils, 'clip_grad_norm_'), "PyTorch gradient clipping not available"

# Test state shapes for bidirectional LSTM
test_lstm = LSTM(10, 20, num_layers=2, bidirectional=True)
test_input = torch.randn(4, 5, 10)
output, (h_state, c_state) = test_lstm(test_input)
assert h_state.shape == (2, 4, 40), f"Expected (2,4,40), got {h_state.shape}"

# Test teacher forcing consistency
train_predictions = model.forward_with_teacher_forcing(...)
val_predictions = validate_with_same_strategy(...)
# Should use similar decoding strategies
```

These fixes will make the NMT implementation more robust, efficient, and production-ready.