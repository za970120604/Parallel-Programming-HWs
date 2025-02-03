# Compiler and flags
CXX = mpic++
CXXFLAGS = -O3 -std=c++11

# Target executable name
TARGET = matmul

# Object files
OBJS = main.o matmul.o

# Default rule to build the target
all: $(TARGET)

# Rule to link object files into the final executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Rule to compile matmul.cc
matmul.o: matmul.cc
	$(CXX) $(CXXFLAGS) -c matmul.cc

# Rule to compile main.cc
main.o: main.cc
	$(CXX) $(CXXFLAGS) -c main.cc

# Clean rule to remove object files and the executable
clean:
	rm -f $(OBJS) $(TARGET)
