#include "../matrix/matrix.h"
#include <memory>

class Layer
{
public:
    Matrix WeightMatrix;
    Matrix BiasMatrix;
    Matrix WeightedSum;
public:
    Layer(unsigned int inputNeurons, unsigned int outputNeurons);
    Matrix UpdateLayer(const Matrix& input);
    void SaveLayer(std::ofstream& outfile) const;
    static Layer LoadLayer(std::ifstream& infile);
    Layer& operator=(Layer&& layer);
    Layer(Layer&& layer) noexcept;
    Layer(const Layer& layer);
};