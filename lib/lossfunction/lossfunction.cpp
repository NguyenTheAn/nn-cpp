#include "lossfunction.h"

namespace loss{
    double MeanAbsoluteError::GetLoss(const Matrix& prediction, const Matrix& target) //const
    {
        return Matrix::Map(prediction - target, [](double x) { return abs(x); }).Sum();
    }

    Matrix MeanAbsoluteError::GetDerivative(const Matrix& prediction, const Matrix& target) //const
    {
        return Matrix::Map(prediction - target, [](double x) { return x >= 0 ? 1 : -1; });
    }
    
    double MeanSquaredError::GetLoss(const Matrix& prediction, const Matrix& target) //const
    {
        return Matrix::Map(prediction - target, [](double x) { return x*x; }).Sum() / (target.GetWidth() * target.GetHeight());
    }

    Matrix MeanSquaredError::GetDerivative(const Matrix& prediction, const Matrix& target) //const
    {
        return (prediction - target) * (2.0 / (target.GetWidth() * target.GetHeight()));
    }

    double CrossEntropy::GetLoss(const Matrix& prediction, const Matrix& target) //const
    {
        std::vector<double> predictionVector = prediction.GetColumnVector();
        std::vector<double> targetVector = target.GetColumnVector();
        double sum = 0.0;
        std::vector<double>::iterator tIt = targetVector.begin();
        for (std::vector<double>::iterator pIt = predictionVector.begin(); pIt != predictionVector.end(); ++pIt, ++tIt)
        {
            double value = -*tIt*log(*pIt) - (1 - *tIt)*log(1 - *pIt);
            if (!std::isnan(value)) sum += value;
        }
        return sum;
    }

    Matrix CrossEntropy::GetDerivative(const Matrix& prediction, const Matrix& target) //const
    {
        return prediction - target;
    }
}