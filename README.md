# CUDA Neural Network Engine

A from-scratch C++ neural network engine with custom matrix operations, dense layers, activation functions, loss computation, training examples, CPU benchmarking, and CUDA-accelerated matrix multiplication.

This project is built to understand how neural networks work under the hood, from basic matrix math and backpropagation to GPU parallelism with CUDA.

---

## Overview

Most machine learning libraries hide the low-level details behind high-level APIs. This project takes the opposite approach: it implements the core pieces manually in C++.

The engine currently includes:

* A custom `Matrix` class
* Matrix addition, subtraction, scalar multiplication, matrix multiplication, transpose, and randomization
* Dense layer forward and backward propagation
* ReLU, Sigmoid, and Softmax activation functions
* Mean Squared Error loss
* Parameter updates with gradient descent
* Training examples for simple regression, batch regression, two-layer learning, and XOR
* CPU matrix multiplication benchmarking
* CUDA matrix multiplication correctness testing
* Reusable CUDA matrix multiplication module
* Matrix-based CPU vs CUDA benchmark comparison

---

## Why I Built This

I wanted to understand neural networks beyond just calling a framework function. This project helped me learn:

* How tensors and matrices are stored in memory
* How forward propagation passes data through layers
* How gradients flow backward through a network
* How weight and bias gradients are computed
* How matrix multiplication becomes a performance bottleneck
* How CUDA can parallelize matrix multiplication by assigning GPU threads to output cells
* How to connect GPU code back to a custom C++ engine abstraction

The CUDA portion was added to explore how the same matrix multiplication workload performs on a GPU compared to a CPU implementation.

---

## Project Structure

```text
CUDA-Neural-Network-Engine/
├── apps/
│   ├── train_simple.cpp
│   ├── train_linear.cpp
│   ├── train_two_layer.cpp
│   └── train_xor.cpp
├── benchmarks/
│   ├── benchmark_cpu_matrix_mult.cpp
│   ├── benchmark_cuda_matrix_mult.cu
│   └── benchmark_matrix_cpu_vs_cuda.cu
├── include/
│   └── neural_engine/
│       ├── activations/
│       ├── cuda/
│       │   └── CUDA_MatrixMult.hpp
│       ├── layers/
│       ├── loss/
│       └── matrix/
├── src/
│   ├── activations/
│   ├── cuda/
│   │   └── CUDA_MatrixMult.cu
│   ├── layers/
│   ├── loss/
│   └── matrix/
├── tests/
└── CMakeLists.txt
```

---

## Core Features

### Custom Matrix Class

The `Matrix` class stores matrix data in a flat `std::vector<double>` and supports common operations used in neural networks.

Implemented operations include:

* Element access with bounds checking
* Matrix addition
* Matrix subtraction
* Scalar multiplication
* Matrix multiplication
* Transpose
* Random initialization
* Row-vector broadcasting for bias addition
* Raw contiguous memory access through `rawData()` for CUDA integration

The matrix multiplication implementation is intentionally written from scratch to make the CPU and CUDA performance comparison meaningful.

---

### Dense Layer

The dense layer implements both forward and backward propagation.

Forward pass:

```text
output = input * weights + bias
```

Backward pass computes:

```text
dW = input^T * outputGradient
db = column-wise sum of outputGradient
dInput = outputGradient * weights^T
```

The layer also supports gradient descent parameter updates:

```text
parameter = parameter - learningRate * gradient
```

---

### Activation Functions

Implemented activation functions:

* ReLU
* Sigmoid
* Softmax

ReLU and Sigmoid include backward propagation support. Sigmoid caches its forward output so the derivative can be computed efficiently:

```text
sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
```

---

### Loss Function

The engine currently includes Mean Squared Error loss.

Forward:

```text
MSE = average((prediction - target)^2)
```

Backward:

```text
dLoss/dPrediction = 2 * (prediction - target) / numberOfValues
```

---

## Training Examples

### Simple Regression

A one-layer model learns a basic mapping:

```text
1 -> 2
```

This validates that the engine can perform forward propagation, compute loss, backpropagate gradients, and update parameters.

---

### Batch Linear Regression

A one-layer model learns:

```text
1 -> 2
2 -> 4
3 -> 6
4 -> 8
```

This confirms that the engine supports batched inputs where each row represents a training example.

---

### Two-Layer Network

A two-layer model verifies multi-layer backpropagation:

```text
Input
  -> Dense
  -> Sigmoid
  -> Dense
  -> Prediction
```

This demonstrates that gradients can flow backward through multiple layers.

---

### XOR Learning

The XOR problem is not linearly separable, so it requires a hidden layer and non-linearity.

Dataset:

```text
[0, 0] -> 0
[0, 1] -> 1
[1, 0] -> 1
[1, 1] -> 0
```

The model uses:

```text
Dense(2, 8)
Sigmoid
Dense(8, 1)
Sigmoid
```

Final predictions after training were close to:

```text
0.0077
0.9925
0.9926
0.0083
```

This shows that the engine can learn a non-linear function from scratch.

---

## CPU Matrix Multiplication Benchmark

The CPU benchmark measures the custom `Matrix::matrixMultiplication()` implementation.

Average CPU timing over 5 trials:

| Matrix Size | Average CPU Time |
| ----------- | ---------------: |
| 128 x 128   |          31.6 ms |
| 256 x 256   |         278.6 ms |
| 512 x 512   |          3647 ms |

Matrix multiplication scales roughly with `O(n^3)` work for square matrices, so runtime grows quickly as matrix size increases.

---

## CUDA Matrix Multiplication

The CUDA implementation assigns one GPU thread to compute one output cell of the result matrix.

For:

```text
C = A * B
```

each CUDA thread computes:

```text
C(row, col)
```

using the dot product of one row from `A` and one column from `B`.

The CUDA kernel uses a 2D block and grid layout:

```cpp
dim3 blockSize(16, 16);
```

Each block computes a `16 x 16` tile of the output matrix.

---

## CUDA Correctness Test

The CUDA implementation was first tested on a small known example:

```text
A:
[1 2]
[3 4]

B:
[5 6]
[7 8]
```

Expected result:

```text
C:
[19 22]
[43 50]
```

Observed result:

```text
19 22
43 50
```

This confirmed that CUDA memory transfer, kernel launch, indexing logic, and result copy-back were working correctly.

---

## Matrix-Based CUDA Benchmark

The CUDA benchmark uses the engine's actual `Matrix` objects instead of standalone raw arrays. The benchmark creates randomized `Matrix` inputs, exposes their contiguous memory through `rawData()`, and passes those raw pointers to the CUDA matrix multiplication function.

This connects the CUDA implementation back to the engine design:

```text
Matrix object -> rawData() -> CUDA kernel -> result written back into Matrix
```

Average CUDA timing over 5 trials on a Tesla T4 GPU in Google Colab:

| Matrix Size | Average CUDA Time |
| ----------- | ----------------: |
| 128 x 128   |       0.469824 ms |
| 256 x 256   |       0.962016 ms |
| 512 x 512   |        4.02381 ms |

---

## CPU vs Matrix-Based CUDA Comparison

The final comparison benchmark uses the same `Matrix` inputs for both the CPU and CUDA paths.

CPU path:

```text
A.matrixMultiplication(B)
```

CUDA path:

```text
matrixMultiplicationCUDA(A.rawData(), B.rawData(), C.rawData(), ...)
```

Average timing over 5 trials:

| Matrix Size | Average CPU Time | Average CUDA Time | Approx. Speedup |
| ----------- | ---------------: | ----------------: | --------------: |
| 128 x 128   |          31.6 ms |       0.378406 ms |          ~83.5x |
| 256 x 256   |         278.6 ms |       0.939949 ms |         ~296.4x |
| 512 x 512   |          3647 ms |         5.1779 ms |         ~704.3x |

These results show the benefit of moving matrix multiplication from a single-threaded CPU implementation to a CUDA kernel where each GPU thread computes one output cell.

The current CUDA implementation uses a simple global-memory kernel. Future work includes shared-memory tiling and deeper integration into the neural network training path.

---

## Build Instructions

### Build the C++ Engine

From the project root:

```bash
cmake -B build -G "MinGW Makefiles"
cmake --build build
```

Run tests:

```bash
ctest --test-dir build --output-on-failure
```

---

## Run Training Examples

```bash
./build/apps/train_simple
./build/apps/train_linear
./build/apps/train_two_layer
./build/apps/train_xor
```

On Windows PowerShell:

```powershell
.\build\apps\train_simple.exe
.\build\apps\train_linear.exe
.\build\apps\train_two_layer.exe
.\build\apps\train_xor.exe
```

---

## Run CPU Benchmark

```bash
./build/benchmarks/benchmark_cpu_matrix_mult
```

On Windows PowerShell:

```powershell
.\build\benchmarks\benchmark_cpu_matrix_mult.exe
```

---

## Run Matrix-Based CUDA Benchmark

The CUDA benchmark requires:

* NVIDIA GPU
* NVIDIA driver
* CUDA Toolkit
* `nvcc`

Compile with:

```bash
nvcc -I include src/cuda/CUDA_MatrixMult.cu src/matrix/Matrix.cpp benchmarks/benchmark_cuda_matrix_mult.cu -o benchmark_cuda_matrix_mult
```

Run:

```bash
./benchmark_cuda_matrix_mult
```

Example output:

```text
Matrix object CUDA benchmark:
128 x 128 | Average CUDA time: 0.469824 ms | Trials: 5
256 x 256 | Average CUDA time: 0.962016 ms | Trials: 5
512 x 512 | Average CUDA time: 4.02381 ms | Trials: 5
```

---

## Run CPU vs CUDA Benchmark

Compile with:

```bash
nvcc -I include src/cuda/CUDA_MatrixMult.cu src/matrix/Matrix.cpp benchmarks/benchmark_matrix_cpu_vs_cuda.cu -o benchmark_matrix_cpu_vs_cuda
```

Run:

```bash
./benchmark_matrix_cpu_vs_cuda
```

Example output:

```text
CPU vs CUDA Matrix multiplication benchmark:
128 x 128 | Average CPU time: 31.6 ms | Average CUDA time: 0.378406 ms | Speedup: 83.5081x | Trials: 5
256 x 256 | Average CPU time: 278.6 ms | Average CUDA time: 0.939949 ms | Speedup: 296.399x | Trials: 5
512 x 512 | Average CPU time: 3647 ms | Average CUDA time: 5.1779 ms | Speedup: 704.339x | Trials: 5
```

---

## Testing

The project uses GoogleTest for unit testing.

Test coverage includes:

* Matrix construction
* Matrix element access
* Matrix operations
* Dense layer forward pass
* Dense layer backward pass
* Gradient shape checks
* Bias gradient computation
* Parameter updates
* Activation functions
* Loss function behavior

Run tests with:

```bash
ctest --test-dir build --output-on-failure
```

---

## Current Limitations

This project is still under active development.

Current limitations:

* CUDA matrix multiplication is not yet integrated into the neural network training loop
* Softmax backward and cross-entropy loss are not fully implemented yet
* No MNIST or image dataset training pipeline yet
* CUDA implementation currently uses a simple global-memory kernel, not shared-memory tiling
* CUDA benchmarks are currently compiled manually with `nvcc` instead of through the main CMake build

---

## Future Work

Planned improvements:

* Add shared-memory tiled CUDA matrix multiplication
* Integrate CUDA matrix multiplication into dense layer training
* Add Softmax + Cross Entropy loss
* Train on MNIST with a small MLP
* Add model saving/loading
* Add more benchmarks and profiling results
* Add CUDA support to the CMake build system

---

## What I Learned

Through this project, I practiced:

* C++ class design
* Matrix math for neural networks
* Backpropagation implementation
* Unit testing with GoogleTest
* CMake project structure
* CPU benchmarking
* CUDA kernel programming
* Host-device memory transfer
* GPU thread/block/grid indexing
* Performance comparison between CPU and GPU workloads
* Connecting low-level CUDA kernels back to a custom C++ engine abstraction

---

## Tech Stack

* C++17
* CUDA
* CMake
* GoogleTest
* Google Colab Tesla T4 GPU
* Git/GitHub

---

## Status

This project currently supports training small neural networks from scratch and benchmarking CPU vs CUDA matrix multiplication using the engine's custom `Matrix` class.

The next major step is to optimize CUDA matrix multiplication with shared-memory tiling and integrate CUDA acceleration into the neural network training path.
