// Ensures this header file is included only once during compilation.
#pragma once

#include <vector>

class Matrix {
    private:
        // Matrix dimensions.
        // rows = number of rows, cols = number of columns.
        int rows, cols;

        // Stores matrix values in one flat dynamic array using row-major order.
        // Example: a 2 x 3 matrix is stored as:
        // [a00, a01, a02, a10, a11, a12]
        std::vector<double> data;

    public:
        // Constructs a matrix with r rows and c columns.
        // The implementation initializes all entries to 0.0.
        Matrix(int r, int c);

        // Writable element access.
        // Converts a 2D position (r, c) into a 1D index:
        // index = r * number_of_columns + c
        //
        // Returning double& lets callers modify the selected element:
        // A(0, 1) = 5.0;
        double& operator()(int r, int c);

        // Read-only element access.
        // Used when the Matrix object is const and should not be modified.
        double operator()(int r, int c) const;

        // Returns the number of rows in the matrix.
        int rowsCount() const;

        // Returns the number of columns in the matrix.
        int colsCount() const;

        // Returns the element-wise sum of this matrix and parameterMatrix.
        // Both matrices must have the same shape.
        Matrix matrixAddition(const Matrix& parameterMatrix) const;

        // Returns the element-wise difference between this matrix and secondMatrix.
        // Both matrices must have the same shape.
        Matrix matrixSubtraction(const Matrix& secondMatrix) const;

        // Returns a new matrix where every value is multiplied by scalarValue.
        Matrix multiplyByScalar(const double scalarValue) const;

        // Returns the matrix product of this matrix and secondMatrix.
        // Shape rule:
        // If this matrix is m x n, secondMatrix must be n x p.
        // The result will be m x p.
        Matrix matrixMultiplication(const Matrix& secondMatrix) const;

        // Returns the transpose of this matrix.
        // Rows become columns and columns become rows.
        Matrix transpose() const;

        // Creates a matrix with random values between minValue and maxValue.
        // This is useful for initializing neural network weights so neurons
        // start with different values and can learn different patterns.
        static Matrix matrixRandomization(int rows, int cols, double minValue, double maxValue);

        // Adds a row vector to every row of this matrix.
        // This is used for bias broadcasting in dense layers.
        Matrix addRowVector(const Matrix& biasVector) const;

        // Provides read-only access to the matrix's contiguous raw memory.
        // Useful for CUDA and other performance code that needs raw pointers.
        const double* rawData() const;

        // Provides writable access to the matrix's contiguous raw memory.
        // Useful when CUDA or other performance code needs to write into a Matrix.
        double* rawData();
};