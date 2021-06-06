#include "activation.h"

// namespace nn{
	namespace activation {
		void ActivationFunction::SaveActivationFunction(std::ofstream & out) const
		{
			Type type = GetType();
			out.write((char*)&type, sizeof(type));
		}

		Matrix ReLu::Function(Matrix& x)
		{
			return x.Map([](double a) { return a >= 0 ? a : 0; });
		}

		Matrix ReLu::Derivative(Matrix& x)
		{
			return x.Map([](double a) { return a >= 0 ? 1 : 0; });
		}

		Type ReLu::GetType() const
		{
			return RELU;
		}

		Matrix Sigmoid::Function(Matrix& x)
		{
			m_Activation = x.Map([](double a) { return 1 / (1 + exp(-a)); });
			return m_Activation;
		}

		Matrix Sigmoid::Derivative(Matrix& x)
		{
			return m_Activation.Map([](double a) { return a * (1 - a); });
		}
		Type Sigmoid::GetType() const
		{
			return SIGMOID;
		}

		Matrix Softmax::Function(Matrix& x)
		{
			double sum = 0.0;
			Matrix::Map(x, [&sum](double a)
			{
				sum += exp(a); return a;
			});
			std::cout<<x<<std::endl;
			m_Activation = x.Map([sum](double a) { return exp(a) / sum; });
			return m_Activation;
		}

		Matrix Softmax::Derivative(Matrix& x)
		{
			return m_Activation.Map([](double a) { return a*(1 - a); });
		}

		Type Softmax::GetType() const
		{
			return SOFTMAX;
		}
	}

	std::shared_ptr<activation::ActivationFunction> ActivationFunctionFactory::BuildActivationFunction(activation::Type type) {
		switch (type)
		{
		case activation::Type::SIGMOID:
			return std::make_shared<activation::Sigmoid>();
		case activation::Type::RELU:
			return std::make_shared<activation::ReLu>();
		case activation::Type::SOFTMAX:
			return std::make_shared<activation::Softmax>();
		default:
			return nullptr;
		}
	}
// }