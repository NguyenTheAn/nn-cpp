#pragma once
#include "../matrix/matrix.h"

namespace activation
{

    // class ActivationFunction
    // {
    // public:
    //     virtual Matrix Function(Matrix& x) = 0;
    //     virtual Matrix Derivative(Matrix& x) = 0;
    //     virtual void SaveActivationFunction(std::ofstream& out) const;
    // };

    class Sigmoid //: public ActivationFunction
    {
        public:
            Matrix m_Activation;
        public:
            Sigmoid(unsigned int inputNeurons);
            Sigmoid(const Matrix& matrix);
            Matrix Function(Matrix& x); //override;
            Matrix Derivative(Matrix& x); //override;
            void SaveActivation(std::ofstream& outfile) const;
            static Sigmoid LoadActivation(std::ifstream& infile);
    };

    class ReLu //: public ActivationFunction
    {
        public:
            Matrix m_Activation;
        public:
            ReLu(unsigned int inputNeurons);
            ReLu(const Matrix& matrix);
            Matrix Function(Matrix& x); //override;
            Matrix Derivative(Matrix& x); //override;
            void SaveActivation(std::ofstream& outfile) const;
            static ReLu LoadActivation(std::ifstream& infile);
    };

    class Softmax //: public ActivationFunction
    {
        public:
            Matrix m_Activation;
        public:
            Softmax(unsigned int inputNeurons);
            Softmax(const Matrix& matrix);
            Matrix Function(Matrix& x); //override;
            Matrix Derivative(Matrix& x); //override;
            void SaveActivation(std::ofstream& outfile) const;
            static Softmax LoadActivation(std::ifstream& infile);
    };
}