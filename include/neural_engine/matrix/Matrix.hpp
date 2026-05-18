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
    public:
        // Constructor: function inside class that runs automatically 
        // when we create an object
        // We need constructor to make sure our variables don't hold random data'
            //It helps initialize the variables
            //Helps allocate memory
        

        // I am using member initializer lists
        // Instead of creating rows and cols first, then assign r and c into them,
        // create rows with r and c immediately
        Matrix(int r, int c)
            :rows(r), cols(c){
        }
};