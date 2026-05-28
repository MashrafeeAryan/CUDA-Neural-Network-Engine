#include "neural_engine/matrix/Matrix.hpp"
#include <random>
#include <stdexcept>

// Constructor: function inside class that runs automatically 
// when we create an object
// We need constructor to make sure our variables don't hold random data'
    //It helps initialize the variables
    //Helps allocate memory


// I am using member initializer lists
// Instead of creating rows and cols first, then assign r and c into them,
// create rows with r and c immediately
// Initialized data vector with 0.0
Matrix::Matrix(int r, int c)
    :rows(r), cols(c), data(r * c, 0.0){
}

// So this is how our vector will look like
// Matrix(2,3) = [0,0,0,0,0,0]
// Actual Matrix:
//        [ 0,0,0
//          0,0,0]
// Now to access each element of vector we use row
// How the index works
    // A(1,2) You want to access first row second element
    // index = element_searching_row * number of columns + element_searching_column
    // index = 1 * 3 + 2 = 5

//& is used as a reference operator
//double& gives direct access to memory slot that stores the retuned element        

// This is used for write/edit version
double& Matrix::operator()(int r, int c){
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
        throw std::out_of_range("Matrix index out of range");   
    }
    return data[ r * cols + c];
}

//Read only version of operator
double Matrix::operator()(int r, int c) const{
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
        throw std::out_of_range("Matrix index out of range");   
    }
    return data[ r * cols + c];
}
//Returns the number of rows a matrix has
int Matrix::rowsCount() const{
    return rows;
}

//return the number of columns a matrix has
int Matrix::colsCount() const{
    return cols;
}

//Define Matrix Addition
//It is usually called :
// Matrix A(2, 2);
// Matrix B(2, 2);
// A(0, 0) = 1;
// A(0, 1) = 2;
// A(1, 0) = 3;
// A(1, 1) = 4;

// B(0, 0) = 5;
// B(0, 1) = 6;
// B(1, 0) = 7;
// B(1, 1) = 8;
// A.matrixAddition(B)
//For addition the rows and columsn should be exact of both matrixs
Matrix Matrix::matrixAddition(const Matrix& secondMatrix) const {
    //Since the matrixAddition is part of current class, we can access provoate values like rows
    //without using getter function.
    //It simply means: (this)(rows)
    if (rows != secondMatrix.rowsCount() || cols!= secondMatrix.colsCount()){
        throw std::invalid_argument("Matrix dimensions must match for addition");    }
    //Create an empty matrix filled with zeroes
    Matrix results(rows, cols); 
    
    // This loop access each row and column
    // (*this) allows to access the current object
    for (int r=0; r< rows; r++){
        for (int c=0; c< cols; c++){
            results(r, c) = (*this)(r,c) + secondMatrix(r, c);
        }
    }

    return results;
}


//Define MatrixSubtraction
//It is pretty similar to matrix addition
Matrix Matrix::matrixSubtraction(const Matrix& secondMatrix) const{
    if (rows != secondMatrix.rowsCount() || cols != secondMatrix.colsCount()){
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    };
    //defines an empty matrix with rows and columns
    Matrix results(rows, cols);
    for (int r =0; r < rows; r++){
        for (int c=0; c< cols; c++){
            results(r, c) = (*this)(r,c) - secondMatrix(r, c);
        }
    }
    return results;
}

//Define MultiplyByScalar function
Matrix Matrix::multiplyByScalar(const double scalarValue) const{
    //No need for error messages because we are using only one matrix here
    Matrix results(rows, cols);
    for (int r = 0; r < rows; r++){
        for (int c = 0; c< cols; c++){
            results(r, c) = (*this)(r,c) * scalarValue;
        }
    }
    return results;
}

//Define Matrix Multiplication
Matrix Matrix::matrixMultiplication(const Matrix& secondMatrix) const {
    //Set the condition for matrix multiplicaiton where columns of first matrix must equal rows of second matrix
    if (cols != secondMatrix.rowsCount()){
        throw std::invalid_argument("Matrix dimensions must match for multiplication. Number of columsn of first amtrix should equal number of rows of second matrix");
    }

    //The resulting matrix shape should be rows of first matrix * columns of second matrix
    Matrix result(rows, secondMatrix.colsCount());

    // rows is for first matrix
    //The first two loops actually help us figure out which spot in the new result box should this go
    for (int r = 0; r < rows; r++){
        for (int c = 0; c< secondMatrix.colsCount(); c++){
            double sum = 0;
            for (int k =0; k < cols; k++){
                //For the first matrix, the rows will be same column number will move
                //For second matrix, row will move and column number will be same
                sum += (*this)(r,k) * secondMatrix(k,c);
            }
            result(r, c) = sum;
        }

    }
    return result;

}

//Define tranpose function
Matrix Matrix::transpose() const {
    //The goal is to flips rows and columns of the matrices
    // A(r,c) = results(c,r)
    //Define results matrix
    Matrix results(cols, rows);

    for (int r=0; r < rows; r++){
        for (int c=0; c< cols; c++){
            results(c, r) = (*this)(r,c);
        }
    }
    return results;
}

//Define matrixRandomization
Matrix Matrix::matrixRandomization(int rows, int cols, double minValue, double maxValue){
    if (minValue > maxValue) {
        throw std::invalid_argument("minValue cannot be greater than maxValue");
    }

    Matrix results(rows, cols);

    //Gets random seed from computer
    std::random_device rd;

    //Uses the random seed to generate many random numbers
    //mt19937 is random number eingien in C++
    std::mt19937 generator(rd());

    //It uses the generator to produce random values between minValue and maxValue
    std::uniform_real_distribution<double> distribution(minValue, maxValue);

    for (int r=0; r< rows; r++){
        for (int c=0; c<cols; c++){
            results(r, c) = distribution(generator);
        }
    }
    return results;
 
}