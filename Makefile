# Makefile for evol_neuralnet

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20

# Include directories - change the paths if necessary
INCLUDES := -I/usr/local/include/
LIBS := /usr/local/lib/libgalib.a -lga

PYTHON3 = python3
PYTHON = python

# Executable and source files
TARGET = evol_neuralnet
SRCS = main.cpp
BUILD_DIR = build

# Default target
all: check_python install_deps $(BUILD_DIR)/$(TARGET)

# Build target
$(BUILD_DIR)/$(TARGET): $(SRCS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(BUILD_DIR)/$(TARGET) $(SRCS) $(LIBS)

# Check if python3 exists, if not, try python
check_python:
	@if command -v $(PYTHON3) 2>/dev/null; then \
		echo "Using $(PYTHON3)"; \
		PYTHON_EXEC=$(PYTHON3); \
	elif command -v $(PYTHON) 2>/dev/null; then \
		echo "Using $(PYTHON)"; \
		PYTHON_EXEC=$(PYTHON); \
	else \
		echo >&2 "Neither python3 nor python exist"; \
		exit 1; \
	fi

# Install dependencies
install_deps:
	$(PYTHON_EXEC) -m pip install torch torchvision

# Test target
.PHONY: test
test: $(BUILD_DIR)/$(TARGET)
	@cd $(BUILD_DIR) && ./$(TARGET)

# Clean target
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
