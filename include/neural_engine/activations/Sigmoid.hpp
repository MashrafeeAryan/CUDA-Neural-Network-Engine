#pragma once

#include "neural_engine/matrix/Matrix.hpp"

class Sigmoid {
    public:
    Matrix forward(const Matrix& inputMatrix) const;
};