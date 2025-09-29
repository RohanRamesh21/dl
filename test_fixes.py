#!/usr/bin/env python3
"""
Test script to validate the NMT implementation fixes
"""
import torch
import sys
sys.path.append('.')

from correct_implementation import (
    LSTM, Decoder, Linear, Embedding, 
    train_step, EncoderDecoderModel,
    run_validation_tests
)

def test_gradient_clipping_integration():
    """Test gradient clipping integration with a simple model"""
    print("Testing gradient clipping integration with model...")
    
    # Create a simple model
    vocab_size = 100
    embedding_dim = 32
    hidden_size = 64
    
    # Create dummy model components
    embedding = Embedding(vocab_size, embedding_dim)
    lstm = LSTM(embedding_dim, hidden_size, num_layers=1, return_state=True)
    output_layer = Linear(hidden_size, vocab_size)
    
    # Simple forward pass function
    def simple_model(x):
        emb = embedding(x)
        lstm_out, _ = lstm(emb)
        return output_layer(lstm_out[:, -1, :])  # Use last timestep
    
    # Create dummy data
    batch_size = 8
    seq_len = 10
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size,))
    
    # Create optimizer
    all_params = embedding.parameters() + lstm.parameters() + output_layer.parameters()
    optimizer = torch.optim.Adam(all_params, lr=0.001)
    
    # Test forward pass
    outputs = simple_model(x)
    loss = torch.nn.functional.cross_entropy(outputs, targets)
    
    # Test backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Calculate gradients norm before clipping
    total_norm_before = 0
    for param in all_params:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm_before += param_norm.item() ** 2
    total_norm_before = total_norm_before ** (1. / 2)
    
    # Apply gradient clipping
    max_norm = 1.0
    torch.nn.utils.clip_grad_norm_(all_params, max_norm)
    
    # Calculate gradients norm after clipping
    total_norm_after = 0
    for param in all_params:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm_after += param_norm.item() ** 2
    total_norm_after = total_norm_after ** (1. / 2)
    
    print(f"Gradient norm before clipping: {total_norm_before:.4f}")
    print(f"Gradient norm after clipping: {total_norm_after:.4f}")
    print(f"Max allowed norm: {max_norm}")
    
    if total_norm_before > max_norm:
        assert abs(total_norm_after - max_norm) < 1e-5, f"Gradient clipping failed: expected {max_norm}, got {total_norm_after}"
        print("✓ Gradient clipping working correctly (gradients were clipped)")
    else:
        print("✓ Gradient clipping working correctly (no clipping needed)")
    
    optimizer.step()
    print("✓ Model update successful after gradient clipping")
    return True

def main():
    """Run all validation tests"""
    print("=" * 60)
    print("RUNNING NMT IMPLEMENTATION VALIDATION TESTS")
    print("=" * 60)
    
    # Run the comprehensive validation tests
    run_validation_tests()
    
    print("\n" + "=" * 60)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 60)
    
    # Run gradient clipping integration test
    test_gradient_clipping_integration()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nThe following fixes have been implemented and validated:")
    print("1. ✅ Gradient Clipping - Using PyTorch built-in torch.nn.utils.clip_grad_norm_")
    print("2. ✅ Bidirectional LSTM State Concatenation - Proper tensor stacking")
    print("3. ✅ State Projection Logic - Robust handling of all encoder configurations")
    print("4. ✅ Teacher Forcing Consistency - Matching strategies between training and validation")
    print("5. ✅ Validation Tests - Comprehensive testing of all fixes")
    
if __name__ == "__main__":
    main()