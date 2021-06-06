#include "model.h"

void Model::Add(Layer::InputLayer inputLayer){
    this->inputLayer = inputLayer;
}

void Model::Add(Layer::HiddenLayer hidden){
    this->hiddenLayer.push_back(hidden);
}

void Model::Add(Layer::OutputLayer outputLayer){
    this->outputLayer.WeightMatrix = std::move(outputLayer.WeightMatrix);
    this->outputLayer.BiasMatrix = std::move(outputLayer.BiasMatrix);
    this->outputLayer.Activation = std::move(outputLayer.Activation);
    this->outputLayer.ActivationFunction = std::move(outputLayer.ActivationFunction);
    this->outputLayer.WeightedSum = std::move(outputLayer.WeightedSum);
}