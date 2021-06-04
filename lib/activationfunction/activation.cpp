#include "activation.h"

namespace activation
{

	ReLu::ReLu(unsigned int inputNeurons)
		: m_Activation(1, inputNeurons)
	{}

	ReLu::ReLu(const Matrix& matrix)
		: m_Activation(matrix)
	{}

	Matrix ReLu::Function(Matrix& x)
	{
		return x.Map([](double a) { return a >= 0 ? a : 0; });
	}

	Matrix ReLu::Derivative(Matrix& x)
	{
		return x.Map([](double a) { return a >= 0 ? 1 : 0; });
	}

	void ReLu::SaveActivation(std::ofstream & outfile) const
	{
		m_Activation.SaveMatrix(outfile);
	}

	ReLu ReLu::LoadActivation(std::ifstream & infile)
	{
		Matrix m_Activation = Matrix::LoadMatrix(infile);
		ReLu relu(m_Activation);
		relu.m_Activation = std::move(m_Activation);
		return relu;
	}

	Sigmoid::Sigmoid(unsigned int inputNeurons)
		: m_Activation(1, inputNeurons)
	{}

	Sigmoid::Sigmoid(const Matrix& matrix)
		: m_Activation(matrix)
	{}

	Matrix Sigmoid::Function(Matrix& x)
	{
		m_Activation = x.Map([](double a) { return 1 / (1 + exp(-a)); });
		return m_Activation;
	}

	Matrix Sigmoid::Derivative(Matrix& x)
	{
		return m_Activation.Map([](double a) { return a * (1 - a); });
	}

	void Sigmoid::SaveActivation(std::ofstream & outfile) const
	{
		m_Activation.SaveMatrix(outfile);
	}

	Sigmoid Sigmoid::LoadActivation(std::ifstream & infile)
	{
		Matrix m_Activation = Matrix::LoadMatrix(infile);
		Sigmoid sigmoid(m_Activation);
		sigmoid.m_Activation = std::move(m_Activation);
		return sigmoid;
	}

	Softmax::Softmax(unsigned int inputNeurons)
		: m_Activation(1, inputNeurons)
	{}

	Softmax::Softmax(const Matrix& matrix)
		: m_Activation(matrix)
	{}

	Matrix Softmax::Function(Matrix& x)
	{
		double sum = 0.0;
		Matrix::Map(x, [&sum](double a)
		{
			sum += exp(a); return a;
		});
		m_Activation = x.Map([sum](double a) { return exp(a) / sum; });
		return m_Activation;
	}

	Matrix Softmax::Derivative(Matrix& x)
	{
		return m_Activation.Map([](double a) { return a*(1 - a); });
	}

	void Softmax::SaveActivation(std::ofstream & outfile) const
	{
		m_Activation.SaveMatrix(outfile);
	}
	
	Softmax Softmax::LoadActivation(std::ifstream & infile)
	{
		Matrix m_Activation = Matrix::LoadMatrix(infile);
		Softmax softmax(m_Activation);
		softmax.m_Activation = std::move(m_Activation);
		return softmax;
	}
}