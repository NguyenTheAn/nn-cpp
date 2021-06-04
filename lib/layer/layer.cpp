#include "layer.h"

Matrix Layer::UpdateLayer(const Matrix & input)
{
    WeightedSum = WeightMatrix*input + BiasMatrix;
    return WeightedSum;
}

Layer::Layer(unsigned int inputNeurons, unsigned int outputNeurons)
    : WeightMatrix(outputNeurons, inputNeurons),
    BiasMatrix(outputNeurons, 1),
    WeightedSum(outputNeurons, 1)
{}

Layer::Layer(Layer && layer) noexcept : WeightMatrix(std::move(layer.WeightMatrix)), BiasMatrix(std::move(layer.BiasMatrix)),
    WeightedSum(std::move(layer.WeightedSum))
{}

Layer::Layer(const Layer & layer) : WeightMatrix(layer.WeightMatrix), BiasMatrix(layer.BiasMatrix),
    WeightedSum(layer.WeightedSum)
{}

void Layer::SaveLayer(std::ofstream & outfile) const
{
    WeightMatrix.SaveMatrix(outfile);
    BiasMatrix.SaveMatrix(outfile);
}

Layer Layer::LoadLayer(std::ifstream & infile)
{
    Matrix weightMatrix = Matrix::LoadMatrix(infile);
    Matrix biasMatrix = Matrix::LoadMatrix(infile);
    Layer layer(weightMatrix.GetWidth(), weightMatrix.GetHeight());
    layer.WeightMatrix = std::move(weightMatrix);
    layer.BiasMatrix = std::move(biasMatrix);
    return layer;
}
Layer & Layer::operator=(Layer && layer)
{
    WeightMatrix = std::move(layer.WeightMatrix);
    BiasMatrix = std::move(layer.BiasMatrix);
    WeightedSum = std::move(layer.WeightedSum);
    return *this;
}
