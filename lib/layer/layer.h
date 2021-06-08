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

            HiddenLayer();
            HiddenLayer(unsigned int inputNeurons, unsigned int outputNeurons, activation::Type activationFunction);
            void Initialize();
            void SaveHiddenLayer(std::ofstream& outfile) const;
            static HiddenLayer LoadHiddenLayer(std::ifstream& infile);
            HiddenLayer(HiddenLayer&& layer) noexcept;
            HiddenLayer(const HiddenLayer& layer);
            HiddenLayer& operator=(HiddenLayer&& layer);
            HiddenLayer& operator=(const HiddenLayer& matrix);
            Matrix Forward(Matrix & input);
    };

    class InputLayer{
        public:
            unsigned int input_dims;
            Matrix m_Input;
            
            InputLayer();
            InputLayer(const unsigned int input_dims);
            Matrix Forward(Matrix & input);

    };

    class OutputLayer{
        public:
            Matrix WeightMatrix;
            Matrix BiasMatrix;
            std::shared_ptr<activation::ActivationFunction> ActivationFunction;
            Matrix Activation;
            Matrix WeightedSum;

            OutputLayer();
            OutputLayer(unsigned int inputNeurons, unsigned int outputNeurons, activation::Type activationFunction);
            void Initialize();
            void SaveOutputLayer(std::ofstream& outfile) const;
            static OutputLayer LoadOutputLayer(std::ifstream& infile);
            OutputLayer(OutputLayer&& layer) noexcept;
            OutputLayer(const OutputLayer& layer);
            OutputLayer& operator=(OutputLayer&& layer);
            OutputLayer& operator=(const OutputLayer& matrix);
            Matrix Forward(Matrix & input);
    };

}