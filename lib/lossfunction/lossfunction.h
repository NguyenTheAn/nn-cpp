#pragma once
#include "../matrix/matrix.h"
#include <limits>

namespace loss {
    // class LossFunction
    // {
    // public:
    //     virtual double GetLoss(const Matrix& prediction, const Matrix& target) const = 0;
    //     virtual Matrix GetDerivative(const Matrix& prediction, const Matrix& target) const = 0;
    //     // Matrix Backward(Layer& layer, Matrix& error);
    //     // void PropagateError(Layer& layer, Matrix& error) const;
    // };

    class MeanAbsoluteError //: public LossFunction
    {
    public:
        double GetLoss(const Matrix& prediction, const Matrix& target); //const override;
        Matrix GetDerivative(const Matrix& prediction, const Matrix& target); //const override;
    };

    class MeanSquaredError //: public LossFunction
    {
    public:
        double GetLoss(const Matrix& prediction, const Matrix& target); //const override;
        Matrix GetDerivative(const Matrix& prediction, const Matrix& target); //const override;
    };

    class CrossEntropy //: public LossFunction
    {
    public:
        double GetLoss(const Matrix& prediction, const Matrix& target); //const override;
        Matrix GetDerivative(const Matrix& prediction, const Matrix& target); //const override;
    };

    class CategoricalCrossEntropy{
        public:
            double GetLoss(Matrix& prediction, Matrix& target);
            Matrix GetDerivative(Matrix& prediction, Matrix& target);
    };

}