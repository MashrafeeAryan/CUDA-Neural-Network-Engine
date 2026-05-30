add_library(activations
    ReLU.cpp
)

target_include_directories(activations
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(activations
    PUBLIC
        matrix
        
)