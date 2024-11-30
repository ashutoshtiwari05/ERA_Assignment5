.PHONY: env clean train test all init env_update

# Environment variables
CONDA_ENV_NAME := mnist_light
PYTHON := python

# Find conda executable - check common locations
CONDA_ROOT := $(shell echo "$$HOME/miniconda3" || echo "$$HOME/anaconda3")
CONDA := $(CONDA_ROOT)/bin/conda

# Initialize conda
init:
	@echo "Initializing conda..."
	@if [ ! -f "$(CONDA)" ]; then \
		echo "Error: conda not found at $(CONDA)"; \
		echo "Please install Miniconda from: https://docs.conda.io/en/latest/miniconda.html"; \
		exit 1; \
	fi
	@echo "source $(CONDA_ROOT)/etc/profile.d/conda.sh" >> ~/.bashrc
	@echo "Conda initialization added to ~/.bashrc"
	@echo "Please run: source ~/.bashrc"
	@echo "Then try 'make env' again"

# Create/update conda environment
env:
	@if [ ! -f "$(CONDA)" ]; then \
		echo "Error: conda not found. Please run 'make init' first"; \
		exit 1; \
	fi
	@echo "Creating/updating conda environment using $(CONDA)..."
	source $(CONDA_ROOT)/etc/profile.d/conda.sh && \
	"$(CONDA)" env remove -n $(CONDA_ENV_NAME) --yes 2>/dev/null || true && \
	"$(CONDA)" env create -f environment.yaml && \
	conda activate $(CONDA_ENV_NAME)
	@echo "Conda environment '$(CONDA_ENV_NAME)' is ready."

# Update conda environment
env_update:
	@echo "Updating conda environment..."
	source $(CONDA_ROOT)/etc/profile.d/conda.sh && \
	"$(CONDA)" env update -f environment.yaml --prune && \
	conda activate $(CONDA_ENV_NAME)
	@echo "Conda environment '$(CONDA_ENV_NAME)' has been updated."

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__
	rm -rf data
	rm -f mnist_model.pth
	@echo "Cleaned."

# Train the model
train:
	@echo "Training model..."
	source $(CONDA_ROOT)/etc/profile.d/conda.sh && \
	conda activate $(CONDA_ENV_NAME) && \
	$(PYTHON) mnist_model.py

# Run tests
test:
	@echo "Running tests..."
	source $(CONDA_ROOT)/etc/profile.d/conda.sh && \
	conda activate $(CONDA_ENV_NAME) && \
	$(PYTHON) test_model.py

# Run complete workflow
all: clean env train test

# Help command
help:
	@echo "Available commands:"
	@echo "  make init       - Initialize conda in your shell (run this first)"
	@echo "  make env        - Create/update conda environment (full rebuild)"
	@echo "  make env_update - Update existing environment (faster than full rebuild)"
	@echo "  make clean      - Clean generated files"
	@echo "  make train      - Train the model"
	@echo "  make test       - Run tests"
	@echo "  make all        - Run complete workflow (clean, env, train, test)"
	@echo ""
	@echo "First time setup:"
	@echo "  1. Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html"
	@echo "  2. Run 'make init'"
	@echo "  3. Run 'source ~/.bashrc'"
	@echo "  4. Run 'make env'"
	@echo ""
	@echo "For updating existing environment:"
	@echo "  Run 'make env_update'"