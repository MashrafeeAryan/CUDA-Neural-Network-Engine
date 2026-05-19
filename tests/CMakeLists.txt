#Module in CMake that allows CMake to browse internet and get neccessary files
include (FetchContent)

#It shows what we want to download and then we give it a link to download from this place
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

#Triggers CMake to go and download the zip file and add it the build
FetchContent_MakeAvailable(googletest)


#This turn on CMake's built-in testing engine so it can excute the files
enable_testing();

#Compiles test_matrix.cpp and converts it into e
add_executable(
    matrix/test_matrix.cpp
)

target_link_libraries(matrix_tests
    PRIVATE
        matrix
        gtest_main
)