add_library(loss
    MSE_Loss.cpp
)

target_include_directories(loss
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(loss
    PUBLIC
        matrix
        
)