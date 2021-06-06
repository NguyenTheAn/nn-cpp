#include <iostream>
#include <fstream>
#include "lib/matrix/matrix.cpp"
#include "lib/activationfunction/activation.cpp"
#include "lib/lossfunction/lossfunction.cpp"
#include "lib/layer/layer.cpp"

#define print(x) std::cout << x << std::endl
#define pii std::pair<unsigned int, unsigned int>
#define vb std::vector<double>

int main(){
    Layer::InputLayer inputLayer(1024);
    std::vector <Layer::HiddenLayer> hiddenLayer;
    hiddenLayer.push_back(Layer::HiddenLayer(1024, 512, activation::Type::RELU));
    hiddenLayer.push_back(Layer::HiddenLayer(512, 128, activation::Type::RELU));
    hiddenLayer.push_back(Layer::HiddenLayer(128, 32, activation::Type::RELU));
    Layer::OutputLayer outputLayer(32, 10, activation::Type::SOFTMAX);
    for (Layer::HiddenLayer hidden : hiddenLayer){
        hidden.Initialize();
    }
    outputLayer.Initialize();
    vb data;
    for (int i = 0; i<1024; i++){
        data.push_back(i);
    }
    Matrix input(1, 1024);
    input.m_Matrix = data;
    Matrix x = inputLayer(input.Transpose());
    for (Layer::HiddenLayer hidden : hiddenLayer){
        x = hidden(x);
    }
    Matrix output = outputLayer(x);
    print(output);
    return 0;
}