add_executable(neural_engine_app
    main.cpp
)

add_executable(train_simple
    train_simple.cpp
)
add_executable(train_linear
    train_linear.cpp
)

add_executable(train_two_layer
    train_two_layer.cpp
)

target_link_libraries(neural_engine_app
    PRIVATE
        matrix
        layers
        activations
)

target_link_libraries(train_simple
    PRIVATE
        matrix
        layers
        loss
)


target_link_libraries(train_linear
    PRIVATE
        matrix
        layers
        loss
)

target_link_libraries(train_two_layer
    PRIVATE
        matrix
        layers
        loss
        activations
)