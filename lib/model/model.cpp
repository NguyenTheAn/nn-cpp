#include "model.h"
#include "../layer/layer.h"

#define shape(x) std::cout << x.m_Rows <<" "<< x.m_Columns << std::endl

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

void Model::SaveMode(std::string fileName){
    std::ofstream outfile;
    outfile.open(fileName, std::ios::binary | std::ios::out);
    for (Layer::HiddenLayer hidden : hiddenLayer){
        hidden.SaveHiddenLayer(outfile);
    }
    this->outputLayer.SaveOutputLayer(outfile);
    outfile.close();
}

void Model::LoadModel(std::string fileName){
    std::ifstream infile;
    infile.open(fileName, std::ios::in | std::ios::binary);
    for (int i=0; i<hiddenLayer.size(); i++){
        hiddenLayer[i] = Layer::HiddenLayer::LoadHiddenLayer(infile);
    }
    this->outputLayer = Layer::OutputLayer::LoadOutputLayer(infile);
    infile.close();
}

float Model::Backpropagation(Matrix input, Matrix target, loss::CrossEntropy criterion, float LR){
    Matrix output = Feedforward(input);
    Matrix dL_dZ = criterion.GetDerivative(output, target);
    Matrix dZ_dY = outputLayer.ActivationFunction->Derivative(outputLayer.WeightedSum);
    Matrix dY_dW = hiddenLayer.back().Activation;
    Matrix dL_dB = dL_dZ.DotProduct(dZ_dY);
    Matrix dL_dW = dL_dB * Matrix::Transpose(dY_dW);
    dL_dZ = Matrix::Transpose(outputLayer.WeightMatrix) * dL_dB;

    // update weight and bias
    outputLayer.WeightMatrix -= LR * dL_dW;
    outputLayer.BiasMatrix -= LR * dL_dB;

    
    for (int i = hiddenLayer.size() - 1; i>0; i--){
        dZ_dY = hiddenLayer[i].ActivationFunction->Derivative(hiddenLayer[i].WeightedSum);
        dY_dW = hiddenLayer[i-1].Activation;
        dL_dB = dL_dZ.DotProduct(dZ_dY);
        dL_dW = dL_dB * Matrix::Transpose(dY_dW);
        dL_dZ = Matrix::Transpose(hiddenLayer[i].WeightMatrix) * dL_dB;

        hiddenLayer[i].WeightMatrix -= LR * dL_dW;
        hiddenLayer[i].BiasMatrix -= LR * dL_dB;
    }

    dZ_dY = hiddenLayer[0].ActivationFunction->Derivative(hiddenLayer[0].WeightedSum);
    dY_dW = inputLayer.m_Input;
    dL_dB = dL_dZ.DotProduct(dZ_dY);
    dL_dW = dL_dB * Matrix::Transpose(dY_dW);
    dL_dZ = Matrix::Transpose(hiddenLayer[0].WeightMatrix) * dL_dB;

    hiddenLayer[0].WeightMatrix -= LR * dL_dW;
    hiddenLayer[0].BiasMatrix -= LR * dL_dB;

    return criterion.GetLoss(output, target);
}