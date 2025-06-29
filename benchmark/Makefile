# Makefile for Graph Summarization Benchmark
# ==========================================

# Compiler settings
CXX = g++
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
    # macOS settings
    CXXFLAGS = -O3 -Xpreprocessor -fopenmp -std=c++17 -Wall -Wextra -pthread -I/opt/homebrew/Cellar/libomp/20.1.7/include
    LDFLAGS = -lomp -pthread -L/opt/homebrew/Cellar/libomp/20.1.7/lib
else
    # Linux settings
    CXXFLAGS = -O3 -fopenmp -std=c++17 -Wall -Wextra -pthread
    LDFLAGS = -fopenmp -pthread
endif

# Directories
SRC_DIR = src
BUILD_DIR = build
CONFIG_DIR = config

# Source files
MAGS_SRC = $(SRC_DIR)/mags.cpp
MAGS_DM_SRC = $(SRC_DIR)/mags_dm.cpp
GRAPH_UTILS_SRC = $(SRC_DIR)/graph_utils.cpp
MINHASH_SRC = $(SRC_DIR)/minhash.cpp

# Executable targets
MAGS_BIN = $(SRC_DIR)/mags
MAGS_DM_BIN = $(SRC_DIR)/mags_dm

# Object files
MAGS_OBJ = $(BUILD_DIR)/mags.o $(BUILD_DIR)/graph_utils.o $(BUILD_DIR)/minhash.o
MAGS_DM_OBJ = $(BUILD_DIR)/mags_dm.o $(BUILD_DIR)/graph_utils.o $(BUILD_DIR)/minhash.o

# Default target
all: setup $(MAGS_BIN) $(MAGS_DM_BIN)

# Setup build directory
setup:
	@mkdir -p $(BUILD_DIR)
	@echo "Build directory created"

# Build MAGS executable
$(MAGS_BIN): $(MAGS_OBJ)
	@echo "Linking MAGS executable..."
	$(CXX) $(MAGS_OBJ) -o $@ $(LDFLAGS)
	@echo "MAGS built successfully"

# Build MAGS-DM executable  
$(MAGS_DM_BIN): $(MAGS_DM_OBJ)
	@echo "Linking MAGS-DM executable..."
	$(CXX) $(MAGS_DM_OBJ) -o $@ $(LDFLAGS)
	@echo "MAGS-DM built successfully"

# Compile object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Individual targets
mags: setup $(MAGS_BIN)
mags_dm: setup $(MAGS_DM_BIN)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -f $(MAGS_BIN) $(MAGS_DM_BIN)
	@echo "Clean completed"

# Install dependencies (for macOS)
install-deps-mac:
	@echo "Installing dependencies on macOS..."
	brew install libomp
	@echo "Dependencies installed"

# Install dependencies (for Ubuntu)
install-deps-ubuntu:
	@echo "Installing dependencies on Ubuntu..."
	sudo apt-get update
	sudo apt-get install -y build-essential libomp-dev
	@echo "Dependencies installed"

# Run tests
test: all
	@echo "Running algorithm tests..."
	@python3 test_algorithms.py

# Test query engines with real data
test-queries: 
	@echo "Testing query engines with real data..."
	@python3 test_query_engine.py

# Run full benchmark (test mode)
benchmark-test: all
	@echo "Running benchmark in test mode..."
	@python3 benchmark.py --test

# Run full benchmark
benchmark: all
	@echo "Running full benchmark..."
	@python3 benchmark.py

# Setup Python environment
setup-python:
	@echo "Installing Python dependencies..."
	@pip3 install -r requirements.txt
	@echo "Python dependencies installed"

# Complete setup (C++ + Python)
setup-all: setup-python all
	@echo "Complete setup finished"

# Format code
format:
	@echo "Formatting C++ code..."
	@find $(SRC_DIR) -name "*.cpp" -o -name "*.h" | xargs clang-format -i

# Check code style
lint:
	@echo "Checking code style..."
	@cppcheck --enable=all --std=c++17 $(SRC_DIR)/*.cpp

# Display help
help:
	@echo "Available targets:"
	@echo ""
	@echo "Building:"
	@echo "  all           - Build all executables"
	@echo "  mags          - Build MAGS executable only"
	@echo "  mags_dm       - Build MAGS-DM executable only"
	@echo "  clean         - Clean build artifacts"
	@echo ""
	@echo "Setup:"
	@echo "  setup-python  - Install Python dependencies"
	@echo "  setup-all     - Complete setup (Python + C++)"
	@echo "  install-deps-mac    - Install dependencies on macOS"
	@echo "  install-deps-ubuntu - Install dependencies on Ubuntu"
	@echo ""
	@echo "Testing:"
	@echo "  test          - Run algorithm tests"
	@echo "  test-queries  - Test query engines with real data"
	@echo "  benchmark-test - Run benchmark in test mode"
	@echo "  benchmark     - Run full benchmark"
	@echo ""
	@echo "Code Quality:"
	@echo "  format        - Format C++ code"
	@echo "  lint          - Check code style"
	@echo ""
	@echo "  help          - Display this help message"

# Phony targets
.PHONY: all setup clean install-deps-mac install-deps-ubuntu test test-queries benchmark-test benchmark setup-python setup-all format lint help mags mags_dm 