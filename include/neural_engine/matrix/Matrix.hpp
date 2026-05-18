#include <vector>
class Matrix{
    // We use private variables for data hiding or encapsulation
    // Private vairbales help you set rules to change your variables
        //Example:
            //If deposit value is less than 0, it is not valid
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
        Matrix(int r, int c)
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

        double& operator()(int r, int c){
            return data[ r * cols + c];
        }

};