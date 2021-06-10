#pragma once
#include "../matrix/matrix.h"
#include <limits>

namespace loss {
    class CrossEntropy 
    {
    public:
        double GetLoss(const Matrix& prediction, const Matrix& target); 
        Matrix GetDerivative(const Matrix& prediction, const Matrix& target); 
    };

    class CategoricalCrossEntropy{
        public:
            double GetLoss(Matrix& prediction, Matrix& target);
            Matrix GetDerivative(Matrix& prediction, Matrix& target);
    };

}