# Compiler and flags
CXX := c++
CXXFLAGS := -std=c++23 -Wall -Wextra -O2 -Werror -I. -MMD -MP
LDFLAGS := -pthread
# Add gtest library paths to LDFLAGS for the test target
TEST_LDFLAGS := $(LDFLAGS) -L/usr/local/lib
# Add gtest include paths to CXXFLAGS if they are not in a standard location
# CXXFLAGS += -I/usr/local/include
LDLIBS := -lgtest -lgmock -lgtest_main -lgmock_main

# Directories
BUILD_DIR := out
TARGET := $(BUILD_DIR)/main
TEST_TARGET := $(BUILD_DIR)/test

# --- Source File Definitions ---

# Main entry point
MAIN_SRC := main.cc

# Shared library sources (code used by both main and test targets)
SHARED_SRCS := $(wildcard tensor/*.cc linalg/*.cc)

# Test-specific sources
TEST_SRCS := tests/test.cc tests/tensor/test_tensor.cc tests/math/test_math.cc tests/linalg/test_linalg.cc

# --- Object File Definitions ---

# Create object file lists from the source lists
MAIN_OBJ := $(patsubst %.cc,$(BUILD_DIR)/%.o,$(MAIN_SRC))
SHARED_OBJS := $(patsubst %.cc,$(BUILD_DIR)/%.o,$(SHARED_SRCS))
TEST_OBJS := $(patsubst %.cc,$(BUILD_DIR)/%.o,$(TEST_SRCS))

# --- Dependency Files ---

# All dependency files from all objects
DEPS := $(MAIN_OBJ:.o=.d) $(SHARED_OBJS:.o=.d) $(TEST_OBJS:.o=.d)

# --- Build Rules ---

# Default target
all: $(TARGET)

# Main binary
$(TARGET): $(MAIN_OBJ) $(SHARED_OBJS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Test binary rule
test: $(TEST_TARGET)

# Test binary linking
$(TEST_TARGET): $(TEST_OBJS) $(SHARED_OBJS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(TEST_LDFLAGS) $(LDLIBS)

# Generic compilation rule for all .cc files
# It places the corresponding .o file in the same subdirectory under $(BUILD_DIR)
$(BUILD_DIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean all build artifacts
clean:
	rm -rf $(BUILD_DIR)/*

# Auto-include dependency files to track header changes
-include $(DEPS)
