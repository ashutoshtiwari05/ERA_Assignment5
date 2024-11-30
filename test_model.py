import torch
import pytest
from mnist_model import LightMNIST, train_model, count_parameters
import torch.nn.functional as F
import numpy as np
import time
import gc
import psutil
import os

def test_model_requirements():
    accuracy, param_count = train_model()
    
    # Test parameter count
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"
    
    # Test accuracy
    assert accuracy > 95.0, f"Model accuracy is {accuracy}%, should be greater than 95%"
    
    print("All tests passed!")

def test_model_architecture():
    model = LightMNIST()
    
    # Test 1: Input shape handling
    batch_size = 32
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    assert output.shape == (batch_size, 10), f"Expected output shape (32, 10), got {output.shape}"
    
    # Test 2: Output probability distribution
    probs = torch.exp(output)
    row_sums = torch.sum(probs, dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), "Output probabilities don't sum to 1"
    
    # Test 3: Gradient flow
    loss = F.nll_loss(output, torch.randint(0, 10, (batch_size,)))
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad), "Some gradients are None"
    
    # Test 4: BatchNorm statistics
    assert hasattr(model, 'bn1'), "Model should have batch normalization layers"
    assert model.bn1.running_mean is not None, "BatchNorm running_mean should be initialized"
    assert model.bn1.running_var is not None, "BatchNorm running_var should be initialized"

def test_model_robustness():
    model = LightMNIST()
    batch_size = 16
    
    # Test 1: Handling different batch sizes
    for bs in [1, 4, 16, 32]:
        x = torch.randn(bs, 1, 28, 28)
        output = model(x)
        assert output.shape == (bs, 10), f"Failed for batch size {bs}"
    
    # Test 2: Handling noisy inputs
    x_clean = torch.randn(batch_size, 1, 28, 28)
    x_noisy = x_clean + 0.1 * torch.randn_like(x_clean)
    out_clean = model(x_clean)
    out_noisy = model(x_noisy)
    diff = torch.norm(out_clean - out_noisy)
    assert diff < 5.0, f"Model is too sensitive to noise: diff={diff}"
    
    # Test 3: Dropout consistency
    model.eval()
    x = torch.randn(batch_size, 1, 28, 28)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2), "Dropout should be consistent in eval mode"

def test_model_training():
    model = LightMNIST()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    
    # Test 1: Loss decreases during training
    x = torch.randn(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))
    
    initial_loss = F.nll_loss(model(x), y)
    for _ in range(5):
        optimizer.zero_grad()
        loss = F.nll_loss(model(x), y)
        loss.backward()
        optimizer.step()
    
    final_loss = F.nll_loss(model(x), y)
    assert final_loss < initial_loss, "Loss should decrease during training"
    
    # Test 2: Model state dict
    state_dict = model.state_dict()
    assert all(torch.isfinite(param).all() for param in state_dict.values()), "Model parameters contain NaN or inf"
    
    # Test 3: Optimizer state
    assert all(param_group['lr'] > 0 for param_group in optimizer.param_groups), "Learning rate should be positive"

def test_model_inference():
    model = LightMNIST()
    model.eval()
    
    # Test 1: Deterministic output
    x = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2), "Model should be deterministic in eval mode"
    
    # Test 2: Output range
    batch = torch.randn(100, 1, 28, 28)
    with torch.no_grad():
        outputs = model(batch)
    assert outputs.max() <= 0, "Log probabilities should be <= 0"
    assert outputs.min() >= -float('inf'), "Log probabilities should be finite"
    
    # Test 3: Prediction distribution
    predictions = outputs.argmax(dim=1)
    unique_preds = torch.unique(predictions)
    assert len(unique_preds) > 1, "Model should predict different classes"

def test_model_memory():
    # Test 1: Memory efficiency
    model = LightMNIST()
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024  # KB
    assert param_size < 1024, f"Model parameters use too much memory: {param_size:.2f} KB"
    
    # Test 2: Forward pass memory
    x = torch.randn(32, 1, 28, 28)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    model(x)  # Warmup
    
    # Test 3: Gradient memory
    optimizer = torch.optim.AdamW(model.parameters())
    y = torch.randint(0, 10, (32,))
    loss = F.nll_loss(model(x), y)
    loss.backward()
    optimizer.step()

def test_memory_leaks():
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    model = LightMNIST()
    initial_memory = get_memory_usage()
    
    # Test 1: Memory leak in forward pass
    for _ in range(100):
        x = torch.randn(32, 1, 28, 28)
        with torch.no_grad():
            _ = model(x)
        
        # Force garbage collection
        del x
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    after_forward = get_memory_usage()
    assert (after_forward - initial_memory) < 50, f"Memory leak detected in forward pass: {after_forward - initial_memory:.2f}MB increase"
    
    # Test 2: Memory leak in backward pass
    optimizer = torch.optim.AdamW(model.parameters())
    memory_before_backward = get_memory_usage()
    
    for _ in range(100):
        x = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        output = model(x)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Force cleanup
        del x, y, output, loss
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    after_backward = get_memory_usage()
    assert (after_backward - memory_before_backward) < 50, f"Memory leak detected in backward pass: {after_backward - memory_before_backward:.2f}MB increase"

def test_model_latency():
    model = LightMNIST()
    model.eval()
    
    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            x = torch.randn(1, 1, 28, 28)
            _ = model(x)
    
    # Test 1: Single image inference latency
    latencies = []
    for _ in range(100):
        x = torch.randn(1, 1, 28, 28)
        start_time = time.time()
        with torch.no_grad():
            _ = model(x)
        latencies.append((time.time() - start_time) * 1000)  # Convert to ms
    
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    print(f"\nLatency stats (ms):")
    print(f"Average: {avg_latency:.2f}")
    print(f"P95: {p95_latency:.2f}")
    print(f"P99: {p99_latency:.2f}")
    
    # Assert reasonable latency for CPU inference
    assert avg_latency < 100, f"Average inference latency too high: {avg_latency:.2f}ms"
    assert p99_latency < 200, f"P99 inference latency too high: {p99_latency:.2f}ms"
    
    # Test 2: Batch processing efficiency
    batch_sizes = [1, 4, 8, 16, 32, 64]
    batch_latencies = {}
    
    for bs in batch_sizes:
        x = torch.randn(bs, 1, 28, 28)
        start_time = time.time()
        with torch.no_grad():
            _ = model(x)
        batch_latencies[bs] = (time.time() - start_time) * 1000 / bs  # ms per sample
    
    print("\nBatch processing efficiency (ms/sample):")
    for bs, latency in batch_latencies.items():
        print(f"Batch size {bs}: {latency:.2f}")
        
    # Assert batch processing efficiency
    assert batch_latencies[64] < batch_latencies[1], "Batch processing not efficient"

def test_model_stability():
    model = LightMNIST()
    model.eval()
    
    # Test 1: Numerical stability
    x = torch.randn(1000, 1, 28, 28)
    with torch.no_grad():
        outputs = model(x)
    
    assert not torch.isnan(outputs).any(), "Model produced NaN outputs"
    assert not torch.isinf(outputs).any(), "Model produced Inf outputs"
    
    # Test 2: Memory stability under stress
    initial_memory = psutil.Process(os.getpid()).memory_info().rss
    
    for _ in range(10):
        large_batch = torch.randn(256, 1, 28, 28)
        with torch.no_grad():
            _ = model(large_batch)
        del large_batch
        gc.collect()
    
    final_memory = psutil.Process(os.getpid()).memory_info().rss
    memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB
    
    assert memory_growth < 100, f"Excessive memory growth under stress: {memory_growth:.2f}MB"

if __name__ == "__main__":
    # Add psutil to environment.yaml
    pytest.main([__file__, "-v"]) 