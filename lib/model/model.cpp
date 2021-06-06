#include "model.h"
#include "../layer/layer.h"

Model::Model(){}

void Model::Add(Layer::InputLayer inputLayer){
    this->inputLayer = inputLayer;
}

void Model::Add(Layer::HiddenLayer hidden){
    this->hiddenLayer.push_back(hidden);
}

void Model::Add(Layer::OutputLayer outputLayer){
    this->outputLayer = outputLayer;
}

void Model::Initialize(){
    this->outputLayer.Initialize();
    for (Layer::HiddenLayer hidden : this->hiddenLayer){
        hidden.Initialize();
    }
}

Matrix Model::Feedforward(Matrix input){
    Matrix x = inputLayer(input.Transpose());
    for (Layer::HiddenLayer hidden : hiddenLayer){
        x = hidden(x);
    }
    return outputLayer(x);
}

void Model::SaveMode(std::ofstream& outfile){
    for (Layer::HiddenLayer hidden : hiddenLayer){
        hidden.SaveHiddenLayer(outfile);
    }
    this->outputLayer.SaveOutputLayer(outfile);
}

void Model::LoadModel(std::ifstream& infile){
    for (int i=0; i<hiddenLayer.size(); i++){
        hiddenLayer[i] = Layer::HiddenLayer::LoadHiddenLayer(infile);
    }
    this->outputLayer = Layer::OutputLayer::LoadOutputLayer(infile);
}