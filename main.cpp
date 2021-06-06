#include <iostream>
#include <fstream>
#include "lib/matrix/matrix.cpp"
#include "lib/activationfunction/activation.cpp"
#include "lib/lossfunction/lossfunction.cpp"
#include "lib/layer/layer.cpp"
#include "lib/model/model.cpp"

#define print(x) std::cout << x << std::endl
#define pii std::pair<unsigned int, unsigned int>
#define vb std::vector<double>

Model createModel(unsigned int input_dims, unsigned int num_classes){
    Model model;
    model.Add(Layer::InputLayer(input_dims));
    model.Add(Layer::HiddenLayer(1024, 512, activation::Type::RELU));
    model.Add(Layer::HiddenLayer(512, 128, activation::Type::RELU));
    model.Add(Layer::HiddenLayer(128, 32, activation::Type::RELU));
    model.Add(Layer::OutputLayer(32, num_classes, activation::Type::SOFTMAX));
    model.Initialize();
    return model;
}

int main(){
    Model model = createModel(1024, 10);
    std::ifstream infile;
    infile.open("model.bin", std::ios::in | std::ios::binary);
    model.LoadModel(infile);
    infile.close();

    vb data;
    for (int i = 0; i<1024; i++){
        data.push_back(i);
    }
    Matrix input(1, 1024);
    input.m_Matrix = data;
    Matrix output = model.Feedforward(input);
    print(output);

    // std::ofstream outfile;
    // outfile.open("model.bin", std::ios::binary | std::ios::out);
    // model.SaveMode(outfile);
    // outfile.close();

    // std::ofstream outfile;
    // outfile.open("model.bin", std::ios::binary | std::ios::out);
    // Layer::HiddenLayer layer(1, 2, activation::Type::RELU);
    // layer.SaveHiddenLayer(outfile);
    // outfile.close();

    // std::ifstream infile;
    // infile.open("model.bin", std::ios::in | std::ios::binary);
    // Layer::HiddenLayer layer = Layer::HiddenLayer::LoadHiddenLayer(infile);
    // print(layer.WeightMatrix);
    // infile.close();

    return 0;
}