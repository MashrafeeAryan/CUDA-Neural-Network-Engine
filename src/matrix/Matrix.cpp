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
double& Matrix::operator()(int r, int c){
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
        throw std::out_of_range("Matrix index out of range");   
    }
    return data[ r * cols + c];
}