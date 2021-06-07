#pragma one
#include "../layer/layer.h"
#include "../lossfunction/lossfunction.h"

class Model{
    public:
        Layer::InputLayer inputLayer;
        std::vector <Layer::HiddenLayer> hiddenLayer;
        Layer::OutputLayer outputLayer;

        Model();
        void Add(Layer::InputLayer inputLayer);
        void Add(Layer::HiddenLayer hidden);
        void Add(Layer::OutputLayer outputLayer);
        void Initialize();
        void SaveMode(std::string fileName);
        void LoadModel(std::string fileName);
        Matrix Feedforward(Matrix input);
        float Backpropagation(Matrix input, Matrix target, loss::CrossEntropy criterion, float LR);
};