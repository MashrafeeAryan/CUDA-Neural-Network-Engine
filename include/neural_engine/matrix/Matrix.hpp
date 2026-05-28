//Preprocessor directive to esnure header file is included only once during compiling
#pragma once

#include <vector>
class Matrix{
    // We use private variables for data hiding or encapsulation
    // Private vairbales help you set rules to change your variables
        //Example:
            //If deposit value is less t han 0, it is not valid
            // So, set a rule that says deposit should not be less than 0
    private:
        // declare rows and columsn value.
        int rows, cols;

        //declare vector 
        //vector is a dynamic array
        //std means standard. It is a namespace
        //Namespace is the name of protective box that library's tools are packed inside
        //Scope Resolution Operator - ::
            //Tells compiler where to look for a tool
        std::vector<double> data;

    public:
        // Constructor: function inside class that runs automatically 
        // when we create an object
        // We need constructor to make sure our variables don't hold random data'
            //It helps initialize the variables
            //Helps allocate memory
        

        // I am using member initializer lists
        // Instead of creating rows and cols first, then assign r and c into them,
        // create rows with r and c immediately
        // Initialized data vector with 0.0
        Matrix(int r, int c);     

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
        double& operator()(int r, int c);

        //Read only version of operator
        double operator()(int r, int c) const;

        //Define functions to see the number of rows and columns
        int rowsCount() const;
        int colsCount() const;

        //Matrix addition
        //We add const so it does not modify the object even accidentally
        Matrix matrixAddition(const Matrix& parameterMatrix) const;




        //Matrix Subtraction follows the same idea as matrix addition
        //We declare matrix subtration here
        //We have both read and write operators in place
        //Matrix is the return type
        //We are passing Matrix location in memory to make it faster comapred to copying the matrix again.
        //The const ensures we do not change any objects
        Matrix matrixSubtraction(const Matrix& secondMatrix) const;




        //Scalar Multiplication
        //It is very simply you just multiply the matrix values with a scalar
        //output is scalar
        Matrix multiplyByScalar(const double scalarValue) const;


        //Hard Stuff
        //Matrix Multiplication
        //We are multiplying two matrix and need to make sure both are const
        Matrix matrixMultiplication(const Matrix& secondMatrix) const;
};

