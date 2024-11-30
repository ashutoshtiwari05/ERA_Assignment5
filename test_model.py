import torch
from mnist_model import LightMNIST, train_model

def test_model_requirements():
    accuracy, param_count = train_model()
    
    # Test parameter count
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"
    
    # Test accuracy
    assert accuracy > 95.0, f"Model accuracy is {accuracy}%, should be greater than 95%"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_model_requirements() 