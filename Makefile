# Makefile for evol_neuralnet

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20

# Paths
GALIB_INCLUDE_DIR = /usr/local/include/ga
GALIB_LIB_DIR = /usr/local/lib
PYTHON = python3

# Executable and source files
TARGET = evol_neuralnet
SRCS = main.cpp
BUILD_DIR = build

# Default target
all: check_python install_deps $(BUILD_DIR)/$(TARGET)

# Build target
$(BUILD_DIR)/$(TARGET): $(SRCS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(GALIB_INCLUDE_DIR) -o $(BUILD_DIR)/$(TARGET) $(SRCS) $(GALIB_LIB_DIR)/libgalib.a

# Check if python3 exists
check_python:
	@command -v $(PYTHON) 2>&1 || { echo >&2 "python3 does not exist"; exit 1; }

# Install dependencies
install_deps:
	$(PYTHON) -m pip install torch torchvision

# Test target
.PHONY: test
test: $(BUILD_DIR)/$(TARGET)
	@cd $(BUILD_DIR) && ./$(TARGET)

# Clean target
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
