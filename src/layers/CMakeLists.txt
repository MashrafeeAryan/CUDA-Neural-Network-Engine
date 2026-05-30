add_library(layers
    Dense.cpp
)

target_include_directories(layers
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(layers
    PUBLIC
        matrix
)