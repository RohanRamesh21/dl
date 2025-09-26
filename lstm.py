"""
LSTM implementation from scratch
"""
from tensor import Tensor, zeros, tanh, sigmoid
from nn import Module, Linear, Parameter
import torch


class LSTMCell(Module):
    """Single LSTM cell implementation from scratch"""
    
    def __init__(self, input_size, hidden_size, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        
        # Input-to-hidden weights and biases
        self.W_ii = Linear(input_size, hidden_size, bias=True, device=device)  # input gate
        self.W_if = Linear(input_size, hidden_size, bias=True, device=device)  # forget gate
        self.W_ig = Linear(input_size, hidden_size, bias=True, device=device)  # candidate values
        self.W_io = Linear(input_size, hidden_size, bias=True, device=device)  # output gate
        
        # Hidden-to-hidden weights and biases
        self.W_hi = Linear(hidden_size, hidden_size, bias=False, device=device)  # input gate
        self.W_hf = Linear(hidden_size, hidden_size, bias=False, device=device)  # forget gate
        self.W_hg = Linear(hidden_size, hidden_size, bias=False, device=device)  # candidate values
        self.W_ho = Linear(hidden_size, hidden_size, bias=False, device=device)  # output gate
        
        # Register submodules
        self._modules['W_ii'] = self.W_ii
        self._modules['W_if'] = self.W_if
        self._modules['W_ig'] = self.W_ig
        self._modules['W_io'] = self.W_io
        self._modules['W_hi'] = self.W_hi
        self._modules['W_hf'] = self.W_hf
        self._modules['W_hg'] = self.W_hg
        self._modules['W_ho'] = self.W_ho
    
    def forward(self, x, hidden_state=None):
        """
        Forward pass through LSTM cell
        x: input tensor (batch_size, input_size)
        hidden_state: tuple (h_prev, c_prev) or None
        returns: (h_new, c_new)
        """
        batch_size = x.shape[0]
        
        if hidden_state is None:
            h_prev = zeros(batch_size, self.hidden_size, device=self.device)
            c_prev = zeros(batch_size, self.hidden_size, device=self.device)
        else:
            h_prev, c_prev = hidden_state
        
        # Input gate: i_t = σ(W_ii * x_t + W_hi * h_{t-1} + b_i)
        i_t = sigmoid(self.W_ii(x) + self.W_hi(h_prev))
        
        # Forget gate: f_t = σ(W_if * x_t + W_hf * h_{t-1} + b_f)
        f_t = sigmoid(self.W_if(x) + self.W_hf(h_prev))
        
        # Candidate values: g_t = tanh(W_ig * x_t + W_hg * h_{t-1} + b_g)
        g_t = tanh(self.W_ig(x) + self.W_hg(h_prev))
        
        # Output gate: o_t = σ(W_io * x_t + W_ho * h_{t-1} + b_o)
        o_t = sigmoid(self.W_io(x) + self.W_ho(h_prev))
        
        # Cell state: c_t = f_t * c_{t-1} + i_t * g_t
        c_new = f_t * c_prev + i_t * g_t
        
        # Hidden state: h_t = o_t * tanh(c_t)
        h_new = o_t * tanh(c_new)
        
        return h_new, c_new


class LSTM(Module):
    """Multi-layer LSTM implementation"""
    
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, 
                 return_sequences=True, return_state=False, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.device = device
        
        # Create LSTM layers
        self.layers = []
        for layer in range(num_layers):
            if layer == 0:
                layer_input_size = input_size
            else:
                layer_input_size = hidden_size
            
            lstm_layer = LSTMCell(layer_input_size, hidden_size, device=device)
            self.layers.append(lstm_layer)
            self._modules[f'layer_{layer}'] = lstm_layer
    
    def forward(self, x, initial_state=None):
        """
        Forward pass through LSTM
        x: input tensor 
           - if batch_first=True: (batch_size, seq_len, input_size)
           - if batch_first=False: (seq_len, batch_size, input_size)
        initial_state: list of tuples [(h_0, c_0), ...] for each layer or None
        
        returns:
        - if return_sequences=True: output for all timesteps
        - if return_sequences=False: output for last timestep only
        - if return_state=True: also returns final (h_n, c_n) for each layer
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        # Handle batch_first
        if self.batch_first:
            x = x.transpose(0, 1)  # Convert to (seq_len, batch_size, input_size)
        
        seq_len, batch_size = x.shape[0], x.shape[1]
        
        # Initialize states if not provided
        if initial_state is None:
            initial_state = []
            for _ in range(self.num_layers):
                h_0 = zeros(batch_size, self.hidden_size, device=self.device)
                c_0 = zeros(batch_size, self.hidden_size, device=self.device)
                initial_state.append((h_0, c_0))
        
        # Store all outputs if return_sequences=True
        if self.return_sequences:
            all_outputs = []
        
        # Final states for each layer
        final_states = []
        
        # Process each layer
        layer_input = x
        for layer_idx in range(self.num_layers):
            lstm_layer = self.layers[layer_idx]
            h_prev, c_prev = initial_state[layer_idx]
            
            layer_outputs = []
            
            # Process each timestep
            for t in range(seq_len):
                # Extract timestep input
                input_t = Tensor(layer_input.data[t], requires_grad=layer_input.requires_grad)
                h_t, c_t = lstm_layer(input_t, (h_prev, c_prev))
                layer_outputs.append(h_t)
                h_prev, c_prev = h_t, c_t
            
            # Stack timestep outputs: (seq_len, batch_size, hidden_size)
            layer_output = torch.stack([out.data for out in layer_outputs], dim=0)
            layer_output = Tensor(layer_output, requires_grad=any(out.requires_grad for out in layer_outputs))
            
            # Set up backward function for the stacked output
            def make_backward_fn(outputs_list):
                def backward_fn(grad):
                    # grad: (seq_len, batch_size, hidden_size)
                    for t, out_t in enumerate(outputs_list):
                        if out_t.requires_grad:
                            out_t.backward(grad[t])
                return backward_fn
            
            if layer_output.requires_grad:
                layer_output._backward_fn = make_backward_fn(layer_outputs)
            
            # Store final state
            final_states.append((h_prev, c_prev))
            
            # Prepare input for next layer
            layer_input = layer_output
            
            # Store output from last layer if return_sequences
            if layer_idx == self.num_layers - 1 and self.return_sequences:
                all_outputs = layer_output
        
        # Prepare output
        if self.return_sequences:
            output = all_outputs
        else:
            # Return only the last timestep
            output = layer_input[-1]  # (batch_size, hidden_size)
        
        # Handle batch_first for output
        if self.batch_first and self.return_sequences:
            output = output.transpose(0, 1)  # Convert back to (batch_size, seq_len, hidden_size)
        
        if self.return_state:
            return output, final_states
        else:
            return output


class BiLSTM(Module):
    """Bidirectional LSTM implementation"""
    
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False,
                 return_sequences=True, return_state=False, device='cpu'):
        super().__init__()
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state
        
        # Forward and backward LSTMs
        self.forward_lstm = LSTM(input_size, hidden_size, num_layers, batch_first,
                                return_sequences=True, return_state=return_state, device=device)
        self.backward_lstm = LSTM(input_size, hidden_size, num_layers, batch_first,
                                 return_sequences=True, return_state=return_state, device=device)
        
        self._modules['forward_lstm'] = self.forward_lstm
        self._modules['backward_lstm'] = self.backward_lstm
    
    def forward(self, x, initial_state=None):
        """
        Forward pass through bidirectional LSTM
        """
        # Forward pass
        if self.return_state:
            forward_output, forward_states = self.forward_lstm(x, initial_state)
        else:
            forward_output = self.forward_lstm(x, initial_state)
        
        # Reverse input for backward pass
        if self.batch_first:
            x_reversed = x.flip(dims=[1])  # Reverse along sequence dimension
        else:
            x_reversed = x.flip(dims=[0])  # Reverse along sequence dimension
        
        # Backward pass
        if self.return_state:
            backward_output, backward_states = self.backward_lstm(x_reversed, initial_state)
        else:
            backward_output = self.backward_lstm(x_reversed, initial_state)
        
        # Reverse backward output to align with forward
        if self.batch_first:
            backward_output = backward_output.flip(dims=[1])
        else:
            backward_output = backward_output.flip(dims=[0])
        
        # Concatenate forward and backward outputs
        output = torch.cat([forward_output.data, backward_output.data], dim=-1)
        output = Tensor(output, requires_grad=(forward_output.requires_grad or backward_output.requires_grad))
        
        def backward_fn(grad):
            # Split gradient for forward and backward
            hidden_size = grad.shape[-1] // 2
            grad_forward = grad.data[..., :hidden_size]
            grad_backward = grad.data[..., hidden_size:]
            
            if forward_output.requires_grad:
                forward_output.backward(grad_forward)
            if backward_output.requires_grad:
                # Reverse gradient for backward LSTM
                if self.batch_first:
                    grad_backward = grad_backward.flip(dims=[1])
                else:
                    grad_backward = grad_backward.flip(dims=[0])
                backward_output.backward(grad_backward)
        
        if output.requires_grad:
            output._backward_fn = backward_fn
        
        if not self.return_sequences:
            # Return only last timestep
            if self.batch_first:
                output = output[:, -1, :]  # (batch_size, 2*hidden_size)
            else:
                output = output[-1, :, :]  # (batch_size, 2*hidden_size)
        
        if self.return_state:
            # Combine forward and backward states
            combined_states = []
            for f_state, b_state in zip(forward_states, backward_states):
                f_h, f_c = f_state
                b_h, b_c = b_state
                combined_h = torch.cat([f_h.data, b_h.data], dim=-1)
                combined_c = torch.cat([f_c.data, b_c.data], dim=-1)
                combined_states.append((Tensor(combined_h), Tensor(combined_c)))
            
            return output, combined_states
        else:
            return output