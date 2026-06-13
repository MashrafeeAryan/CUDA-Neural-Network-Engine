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
* CUDA matrix multiplication correctness test and benchmark

---

## Why I Built This

I wanted to understand neural networks beyond just calling a framework function. This project helped me learn:

* How tensors/matrices are stored in memory
* How forward propagation passes data through layers
* How gradients flow backward through a network
* How weight and bias gradients are computed
* How matrix multiplication becomes a performance bottleneck
* How CUDA can parallelize matrix multiplication by assigning GPU threads to output cells

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
│   └── benchmark_cuda_matmul.cu
├── include/
│   ├── activations/
│   ├── layers/
│   ├── loss/
│   └── matrix/
├── src/
│   ├── activations/
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

The CPU benchmark measures the custom C++ matrix multiplication implementation.

Average CPU timing over 5 trials:

| Matrix Size | Average CPU Time |
| ----------- | ---------------: |
| 128 x 128   |            27 ms |
| 256 x 256   |         247.4 ms |
| 512 x 512   |        2419.2 ms |

Matrix multiplication scales roughly with `O(n^3)` work for square matrices, so the runtime grows quickly as matrix size increases.

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

This confirmed that the CUDA memory transfer, kernel launch, indexing logic, and result copy-back were working correctly.

---

## CUDA Benchmark

The CUDA benchmark was run on a Tesla T4 GPU in Google Colab.

Average CUDA timing over 5 trials:

| Matrix Size | Average CUDA Time |
| ----------- | ----------------: |
| 128 x 128   |        0.33943 ms |
| 256 x 256   |       0.873914 ms |
| 512 x 512   |        3.76129 ms |

---

## CPU vs CUDA Comparison

| Matrix Size | CPU Average Time | CUDA Average Time | Approx. Speedup |
| ----------- | ---------------: | ----------------: | --------------: |
| 128 x 128   |            27 ms |        0.33943 ms |            ~80x |
| 256 x 256   |         247.4 ms |       0.873914 ms |           ~283x |
| 512 x 512   |        2419.2 ms |        3.76129 ms |           ~643x |

These results show the benefit of parallelizing matrix multiplication on the GPU.

The current CUDA benchmark uses raw flat arrays to isolate and test the CUDA kernel. A future version will connect the CUDA implementation directly to the custom `Matrix` class for a cleaner engine-level comparison.

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

## Run CUDA Benchmark

The CUDA benchmark requires:

* NVIDIA GPU
* NVIDIA driver
* CUDA Toolkit
* `nvcc`

Compile with:

```bash
nvcc benchmarks/benchmark_cuda_matmul.cu -o benchmark_cuda_matmul
```

Run:

```bash
./benchmark_cuda_matmul
```

Example output:

```text
Correctness test result:
19 22
43 50

CUDA matrix multiplication benchmark:
128 x 128 | Average CUDA time: 0.33943 ms | Trials: 5
256 x 256 | Average CUDA time: 0.873914 ms | Trials: 5
512 x 512 | Average CUDA time: 3.76129 ms | Trials: 5
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

* CUDA matrix multiplication is currently benchmarked as a standalone `.cu` file
* CUDA code is not yet integrated directly into the `Matrix` class
* Softmax backward and cross-entropy loss are not fully implemented yet
* No MNIST or image dataset training pipeline yet
* CUDA implementation currently uses a simple global-memory kernel, not shared-memory tiling

---

## Future Work

Planned improvements:

* Refactor CUDA code into `src/cuda`
* Add a `Matrix`-based CUDA multiplication wrapper
* Compare `Matrix::matrixMultiplication()` against CUDA multiplication directly
* Add shared-memory tiled CUDA matrix multiplication
* Add Softmax + Cross Entropy loss
* Train on MNIST with a small MLP
* Add model saving/loading
* Add more benchmarks and profiling results

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

This project currently supports training small neural networks from scratch and benchmarking CPU vs CUDA matrix multiplication.

The next major step is to refactor CUDA matrix multiplication into reusable engine code and connect it directly to the custom `Matrix` class.
