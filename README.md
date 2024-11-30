# Lightweight MNIST Classifier

This project implements a lightweight neural network for MNIST digit classification that achieves >95% accuracy in just one epoch with less than 25,000 parameters.

## Model Architecture

The model uses a simple CNN architecture:
- 2 Convolutional layers
- 2 Max pooling layers
- 2 Fully connected layers
- Total parameters: ~24,000

## Requirements

- Conda (Miniconda or Anaconda)
- Make (usually pre-installed on Linux/Mac)

## Installation and Setup

1. Install Miniconda (if not already installed):
   - Download from [Miniconda website](https://docs.conda.io/en/latest/miniconda.html)
   - Follow installation instructions for your OS
   - Initialize conda for your shell:
     ```bash
     conda init <your-shell-name>  # e.g., conda init bash or conda init zsh
     ```
   - Restart your terminal

2. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

3. Create and activate the conda environment:
```bash
make env
conda activate mnist_light
```

## Usage

The project includes a Makefile for easy execution of common tasks:

1. Train the model:
```bash
make train
```

2. Run tests:
```bash
make test
```

3. Run complete workflow (clean, setup env, train, and test):
```bash
make all
```

4. Clean generated files:
```bash
make clean
```

5. Show available commands:
```bash
make help
```

## Model Specifications

- Parameters: <25,000
- Accuracy: >95% in 1 epoch
- Training time: ~2-3 minutes on CPU

## CI/CD

The repository includes GitHub Actions that automatically test:
1. Parameter count constraint (<25,000)
2. Accuracy requirement (>95%)

## Project Structure

```
├── mnist_model.py     # Main model and training code
├── test_model.py      # Test script
├── environment.yaml   # Conda environment specification
├── Makefile          # Build automation
├── .github/workflows  # GitHub Actions configuration
└── README.md         # Documentation
```

## License

MIT
