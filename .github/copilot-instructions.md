# Neural Machine Translation Architecture Enhancement

## Current Code Analysis
The existing code implements a basic encoder-decoder with attention architecture using PyTorch tensors from scratch. Now enhance it with the following features:

## Required Enhancements

### 1. Dropout Implementation
**Location**: Add to Linear, LSTM, and Embedding layers
**Requirements**:
- Add `dropout_rate` parameter to all layer constructors
- Implement dropout using `F.dropout()` in forward passes
- Apply dropout after activation functions where appropriate
- Make dropout optional (rate=0.0 means no dropout)

### 2. Batch Normalization
**Location**: Add BatchNorm class and integrate with Linear layers
**Requirements**:
- Create `BatchNorm` class from scratch using PyTorch tensors
- Support 1D and 2D batch normalization
- Add `use_batchnorm` parameter to Linear layers
- Integrate batchnorm after linear transformation before activation

### 3. Bidirectional LSTM
**Location**: Enhance LSTM and Encoder classes
**Requirements**:
- Add `bidirectional` parameter to LSTM constructor
- Modify LSTM to process sequences in both directions
- Concatenate forward and backward outputs
- Handle bidirectional state initialization for decoder

### 4. Multi-layer Support
**Location**: Enhance LSTM to support multiple layers
**Requirements**:
- Add `num_layers` parameter to LSTM
- Implement layer stacking with dropout between layers
- Properly handle hidden states for multi-layer setup

### 5. Configuration System
**Location**: Create separate configuration script/module
**Requirements**:
- Create `config.py` or `model_config.py`
- Implement configuration classes for model architecture and training
- Support JSON/YAML configuration files
- Include all hyperparameters and architectural choices

## Implementation Specifications
- Make sure the training will be done in an ipynb notebook