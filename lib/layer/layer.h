#pragma once
#include "../activationfunction/activation.h"
#include <memory>

namespace Layer{
    class HiddenLayer{
        public:
            Matrix WeightMatrix;
            Matrix BiasMatrix;
            std::shared_ptr<activation::ActivationFunction> ActivationFunction;
            Matrix Activation;
            Matrix WeightedSum;

            HiddenLayer(unsigned int inputNeurons, unsigned int outputNeurons, activation::Type activationFunction);
            // Matrix UpdateActivation(const Matrix& input);
            void Initialize();
            void SaveHiddenLayer(std::ofstream& outfile) const;
            static HiddenLayer LoadHiddenLayer(std::ifstream& infile);
            HiddenLayer(HiddenLayer&& layer) noexcept;
            HiddenLayer(const HiddenLayer& layer);
            HiddenLayer& operator=(HiddenLayer&& layer);
            Matrix& operator()(Matrix & input);
    };

    class InputLayer{
        public:
            unsigned int input_dims;
            Matrix m_Input;
            
            InputLayer(const unsigned int input_dims);
            Matrix& operator()(Matrix & input);

    };

    class OutputLayer{
        public:
            Matrix WeightMatrix;
            Matrix BiasMatrix;
            std::shared_ptr<activation::ActivationFunction> ActivationFunction;
            Matrix Activation;
            Matrix WeightedSum;

            OutputLayer(unsigned int inputNeurons, unsigned int outputNeurons, activation::Type activationFunction);
            void Initialize();
            void SaveOutputLayer(std::ofstream& outfile) const;
            static OutputLayer LoadOutputLayer(std::ifstream& infile);
            OutputLayer(OutputLayer&& layer) noexcept;
            OutputLayer(const OutputLayer& layer);
            OutputLayer& operator=(OutputLayer&& layer);
            Matrix& operator()(Matrix & input);
    };

}