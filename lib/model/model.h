#pragma one
#include "../layer/layer.h"

class Model{
    public:
        Layer::InputLayer inputLayer;
        std::vector <Layer::HiddenLayer> hiddenLayer;
        Layer::OutputLayer outputLayer;

        Model();
        void Add(Layer::InputLayer inputLayer);
        void Add(Layer::HiddenLayer hidden);
        void Add(Layer::OutputLayer outputLayer);
};