# CPP-CUDA-Neural-Network
A from-scratch neural network engine in C++ with CPU parallelism and CUDA-accelerated matrix multiplication.


#Cmake Commands:
Remove-Item -Recurse -Force build
cmake -B build -G "MinGW Makefiles"
cmake --build build
ctest --test-dir build --output-on-failure


 
cmake --build build
ctest --test-dir build --output-on-failure

cmake --build build
.\build\apps\neural_engine_app.exe