NVCC ?= nvcc
TARGET := bin/cuda_vsr
SRC := src/main.cpp src/kernels.cu
INCLUDES := -Iinclude
CXXFLAGS := -O3 -std=c++17 -lineinfo
CUDA_ARCH_FLAGS ?= -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p bin
	$(NVCC) $(CXXFLAGS) $(CUDA_ARCH_FLAGS) $(INCLUDES) -Xcompiler -Wall -o $@ $(SRC)

clean:
	@rm -f $(TARGET)
