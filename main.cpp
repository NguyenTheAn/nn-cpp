#include <iostream>
#include <fstream>
#include "lib/matrix/matrix.cpp"
#include "lib/activationfunction/activation.cpp"
#include "lib/lossfunction/lossfunction.cpp"
#include "lib/layer/layer.cpp"

#define LOG(x) std::cout << x << std::endl

int main(){
    Layer inputlayer(3, 1);
    Layer hidden1(3, 4);
    activation::ReLu relu(4);
    Matrix x = hidden1.UpdateLayer(inputlayer.WeightedSum);
    relu.m_Activation = relu.Function(x);
    LOG(x);
    return 0;
}