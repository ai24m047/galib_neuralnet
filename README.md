# Neural Network Hyperparameter Optimization

## Project Overview

This project implements a genetic algorithm-based approach to automatically optimize neural network hyperparameters using a unique combination of C++ and Python technologies.

### Key Features
- Genetic Algorithm for Hyperparameter Search
- Dynamic Neural Network Architecture
- MNIST Digit Classification (Subset 0-4)
- Configurable Optimization Parameters
- Performance Logging and Tracking

## Prerequisites

### System Requirements
- C++ Compiler with C++17 support (GCC, Clang, or MSVC)
- Python 3.8+
- PyTorch
- Torchvision
- GALib (Genetic Algorithm Library)

### Dependencies Installation

#### C++ Dependencies
- GALib: Download and install from the official source
- Recommended compilation flags: `-std=c++17`

#### Python Dependencies
pip install torch torchvision

## Project Structure
project_root/

├── CMakeLists.txt           # CMake build configuration

├── evol_neuralnet.py        # Python script for training the neural network

├── main.cpp                 # C++ program implementing the genetic algorithm

├── galib/                   # GAlib library directory (must be cloned from GitHub)

├── build/                   # Build directory (generated by CMake)

└── logs/                    # Log files (generated during the genetic algorithm run)

## Detailed Functionality

### Genetic Algorithm (C++)
The C++ component (`main.cpp`) implements a genetic algorithm that:
- Explores neural network hyperparameter space
- Supports dynamic configuration of:
   - Learning rate
   - Dropout rate
   - Batch size
   - Epochs
   - Number of hidden layers
   - Layer sizes
   - Activation functions

#### Configurable Parameters
You can customize the genetic algorithm and neural network search space via command-line arguments:

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--population` | Population size | 15 | 1-1000 |
| `--generations` | Maximum number of generations | 5 | 1-1000 |
| `--mutation` | Mutation probability | 0.2 | 0.0-1.0 |
| `--crossover` | Crossover probability | 0.7 | 0.0-1.0 |
| `--fitness-threshold` | Minimum average fitness for convergence | 0.92 | 0.0-1.0 |
| `--consecutive-gen` | Consecutive generations to meet threshold | 5 | 1-1000 |
| `--min-lr` | Minimum learning rate | 0.0001 | 0.0-1.0 |
| `--max-lr` | Maximum learning rate | 0.1 | 0.0-1.0 |
| `--min-dropout` | Minimum dropout rate | 0.0 | 0.0-1.0 |
| `--max-dropout` | Maximum dropout rate | 0.5 | 0.0-1.0 |
| `--min-batch` | Minimum batch size | 16 | 1-1000 |
| `--max-batch` | Maximum batch size | 128 | 1-1000 |
| `--min-epochs` | Minimum training epochs | 1 | 1-1000 |
| `--max-epochs` | Maximum training epochs | 50 | 1-1000 |
| `--min-layers` | Minimum number of hidden layers | 1 | 1-1000 |
| `--max-layers` | Maximum number of hidden layers | 5 | 1-1000 |
| `--min-layer-size` | Minimum nodes per hidden layer | 16 | 1-1000 |
| `--max-layer-size` | Maximum nodes per hidden layer | 128 | 1-1000 |

### Neural Network (Python)
The Python script (`evol_neuralnet.py`) implements:
- Dynamic neural network architecture
- MNIST digit classification (digits 0-4)
- Fitness score calculation
- Training and evaluation pipeline

## Usage Examples

### Basic Run

Run with default parameters
   ./evol_neuralnet

Advanced configuration example
./evol_neuralnet \
   --population 50 \
   --generations 20 \
   --mutation 0.15 \
   --crossover 0.8 \
   --min-lr 0.0005 \
   --max-lr 0.05

Display all available options
   ./evol_neuralnet --help

## Logging

The system generates detailed logs in the `../logs/` directory:
- `evolution_log_<timestamp>.csv`: Genetic algorithm evolution tracking
- `training_log.txt`: Neural network training details

## Performance Metrics

The optimization targets a combined performance score:
- 70% weight on classification accuracy
- 30% weight on normalized loss
- Final score ranges between 0 and 1

## Technical Details

### Genetic Algorithm Strategy
- Elitism to preserve best solutions
- Dynamic mutation and crossover
- Early stopping based on fitness threshold

### Neural Network Configuration
- Input: MNIST images (28x28 pixels)
- Output: 5 classes (digits 0-4)
- Configurable hidden layers
- Dropout regularization
- Multiple activation functions

## Customization and Extensibility

You can easily extend the project by:
- Adding more activation functions
- Modifying fitness score calculation
- Supporting additional datasets
- Expanding hyperparameter search space

## Troubleshooting

- Ensure all dependencies are correctly installed
- Check that GALib is properly linked
- Verify Python and PyTorch installations
- Review log files for detailed error information

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the project repository.
