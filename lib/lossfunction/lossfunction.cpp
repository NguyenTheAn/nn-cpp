#include "lossfunction.h"

namespace loss{
    double MeanAbsoluteError::GetLoss(const Matrix& prediction, const Matrix& target)
    {
        return Matrix::Map(prediction - target, [](double x) { return abs(x); }).Sum();
    }

    Matrix MeanAbsoluteError::GetDerivative(const Matrix& prediction, const Matrix& target)
    {
        return Matrix::Map(prediction - target, [](double x) { return x >= 0 ? 1 : -1; });
    }
    
    double MeanSquaredError::GetLoss(const Matrix& prediction, const Matrix& target)
    {
        return Matrix::Map(prediction - target, [](double x) { return x*x; }).Sum() / (target.GetWidth() * target.GetHeight());
    }

    Matrix MeanSquaredError::GetDerivative(const Matrix& prediction, const Matrix& target)
    {
        return (prediction - target) * (2.0 / (target.GetWidth() * target.GetHeight()));
    }

    double CrossEntropy::GetLoss(const Matrix& prediction, const Matrix& target)
    {
        std::vector<double> predictionVector = prediction.GetColumnVector();
        std::vector<double> targetVector = target.GetColumnVector();
        double sum = 0.0;
        std::vector<double>::iterator tIt = targetVector.begin();
        for (std::vector<double>::iterator pIt = predictionVector.begin(); pIt != predictionVector.end(); ++pIt, ++tIt)
        {
            double value = -*tIt*log(*pIt) - (1 - *tIt)*log(1 - *pIt);
            if (std::isinf(value) || std::isnan(value)) value = std::numeric_limits<int>::max()*1.0;
            sum += value;
        }
        return sum;
    }

    Matrix CrossEntropy::GetDerivative(const Matrix& prediction, const Matrix& target)
    {
        return prediction - target;
    }

    double CategoricalCrossEntropy::GetLoss(Matrix& prediction, Matrix& target){
        std::vector<double> predictionVector = prediction.GetColumnVector();
        std::vector<double> targetVector = target.GetColumnVector();
        double sum = 0.0;
        std::vector<double>::iterator tIt = targetVector.begin();
        for (std::vector<double>::iterator pIt = predictionVector.begin(); pIt != predictionVector.end(); ++pIt, ++tIt)
        {
            double value = -*tIt*log(*pIt);
            if (std::isinf(value) || std::isnan(value)) value = std::numeric_limits<int>::max()*1.0;
            sum += value;
        }
        return sum;
    }

    Matrix CategoricalCrossEntropy::GetDerivative(Matrix& prediction, Matrix& target){
        return prediction - target;
    }
}