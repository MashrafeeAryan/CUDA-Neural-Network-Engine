#include "neural_engine/matrix/Matrix.hpp"

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