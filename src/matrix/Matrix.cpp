#include "neural_engine/matrix/Matrix.hpp"

#include <random>
#include <stdexcept>

// Creates a rows x cols matrix and initializes every element to 0.0.
// Matrix values are stored in one flat vector using row-major order.
Matrix::Matrix(int r, int c)
    : rows(r), cols(c), data(r * c, 0.0) {
}

// Writable element access.
// Converts a 2D matrix position (r, c) into a 1D vector index:
// index = r * cols + c
double& Matrix::operator()(int r, int c) {
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
        throw std::out_of_range("Matrix index out of range");   
    }

    return data[r * cols + c];
}

// Read-only element access for const Matrix objects.
// This allows safe access without modifying the matrix.
double Matrix::operator()(int r, int c) const {
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
        throw std::out_of_range("Matrix index out of range");   
    }

    return data[r * cols + c];
}

// Returns the number of rows in the matrix.
int Matrix::rowsCount() const {
    return rows;
}

// Returns the number of columns in the matrix.
int Matrix::colsCount() const {
    return cols;
}

// Performs element-wise matrix addition.
// Both matrices must have the same shape.
Matrix Matrix::matrixAddition(const Matrix& secondMatrix) const {
    if (rows != secondMatrix.rowsCount() || cols != secondMatrix.colsCount()) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    Matrix results(rows, cols); 
    
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            results(r, c) = (*this)(r, c) + secondMatrix(r, c);
        }
    }

    return results;
}

// Performs element-wise matrix subtraction.
// Both matrices must have the same shape.
Matrix Matrix::matrixSubtraction(const Matrix& secondMatrix) const {
    if (rows != secondMatrix.rowsCount() || cols != secondMatrix.colsCount()) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }

    Matrix results(rows, cols);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            results(r, c) = (*this)(r, c) - secondMatrix(r, c);
        }
    }

    return results;
}

// Multiplies every matrix element by scalarValue.
Matrix Matrix::multiplyByScalar(const double scalarValue) const {
    Matrix results(rows, cols);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            results(r, c) = (*this)(r, c) * scalarValue;
        }
    }

    return results;
}

// Performs standard matrix multiplication.
// Shape rule:
// If this matrix is rows x cols, secondMatrix must be cols x secondMatrix.colsCount().
// The result shape is rows x secondMatrix.colsCount().
Matrix Matrix::matrixMultiplication(const Matrix& secondMatrix) const {
    if (cols != secondMatrix.rowsCount()) {
        throw std::invalid_argument(
            "Matrix dimensions must match for multiplication. Number of columns of first matrix should equal number of rows of second matrix"
        );
    }

    Matrix result(rows, secondMatrix.colsCount());

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < secondMatrix.colsCount(); c++) {
            double sum = 0;

            // Dot product between row r of this matrix and column c of secondMatrix.
            for (int k = 0; k < cols; k++) {
                sum += (*this)(r, k) * secondMatrix(k, c);
            }

            result(r, c) = sum;
        }
    }

    return result;
}

// Returns the transpose of the matrix.
// Element at (r, c) becomes element at (c, r).
Matrix Matrix::transpose() const {
    Matrix results(cols, rows);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            results(c, r) = (*this)(r, c);
        }
    }

    return results;
}

// Creates a matrix with random values between minValue and maxValue.
// This is useful for neural network weight initialization, where starting
// every weight at the same value can prevent neurons from learning different patterns.
Matrix Matrix::matrixRandomization(int rows, int cols, double minValue, double maxValue) {
    if (minValue > maxValue) {
        throw std::invalid_argument("minValue cannot be greater than maxValue");
    }

    Matrix results(rows, cols);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> distribution(minValue, maxValue);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            results(r, c) = distribution(generator);
        }
    }

    return results;
}

// Adds a row vector to every row of this matrix.
// Used for bias broadcasting in dense layers.
// Expected bias shape: 1 x number_of_columns.
Matrix Matrix::addRowVector(const Matrix& biasVector) const {
    if (cols != biasVector.colsCount() || biasVector.rowsCount() != 1) {
        throw std::invalid_argument(
            "Bias vector must have shape 1 x number_of_columns"
        );
    }

    Matrix result(rows, cols);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            result(r, c) = (*this)(r, c) + biasVector(0, c);
        } 
    }

    return result;
}

// Provides read-only access to the matrix's contiguous raw memory.
// This is useful for CUDA and other performance code that requires raw pointers.
const double* Matrix::rawData() const {
    return data.data();
}

// Provides writable access to the matrix's contiguous raw memory.
// This allows CUDA or other performance code to write results directly into a Matrix.
double* Matrix::rawData() {
    return data.data();
}