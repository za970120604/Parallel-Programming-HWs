CC = g++
CFLAGS = -pthread -march=native -mavx2 -O2

TARGET = pi.out
OBJ = pi.o

INCLUDE_DIR = ./include
SIMD_HEADER = $(INCLUDE_DIR)/SIMDInstructionSet.h
RNG_HEADER = $(INCLUDE_DIR)/Xoshiro256Plus.h

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ)

$(OBJ): pi.cpp $(SIMD_HEADER) $(RNG_HEADER)
	$(CC) $(CFLAGS) -c pi.cpp

clean:
	rm -f $(OBJ) $(TARGET)
