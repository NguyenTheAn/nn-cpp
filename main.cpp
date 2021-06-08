#include <iostream>
#include <fstream>
#include "lib/matrix/matrix.cpp"
#include "lib/activationfunction/activation.cpp"
#include "lib/lossfunction/lossfunction.cpp"
#include "lib/layer/layer.cpp"
#include "lib/model/model.cpp"

#define shape(x) std::cout << x.m_Rows <<" "<< x.m_Columns << std::endl
#define print(x) std::cout << x << std::endl
#define pii std::pair<unsigned int, unsigned int>
#define vb std::vector<double>


int EPOCHS = 10;
float LR = 0.001;
int BATCH_SIZE = 32;

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

Matrix load_data(std::string data_path, int rows, int cols){
    std::ifstream inFile;
    inFile.open(data_path, std::ios::in);
    if (!inFile) {
        std::cerr << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }

    std::vector<double> data(0);
    for (int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            int pixel;
            inFile >> pixel;
            data.push_back(pixel);
        }
    }
    Matrix mat = Matrix(data, rows, cols);
    inFile.close();
    return mat;
}

int main(){
    Model model = createModel(1024, 10);
    // model.LoadModel("model.bin");
    loss::CategoricalCrossEntropy criterion;
    Matrix mat_train_data = load_data("../data/train_data.txt", 1934, 1024);
    Matrix mat_train_label = load_data("../data/train_label.txt", 1934, 10);

    // int i = 0;
    // Matrix input(mat_train_data.GetRow(i));
    // Matrix label(mat_train_label.GetRow(i));
    // model.Backpropagation(Matrix::Transpose(input), Matrix::Transpose(label), criterion, LR);
    // print(model.outputLayer.WeightMatrix);
    // Matrix output = model.Feedforward(input);
    // print(criterion.GetLoss(output, label));

    std::ofstream outFile;
    outFile.open("loss.txt", std::ios_base::out);
    for (int e=0; e<EPOCHS; e++){
        int barLength = 60;
        int pos = (1934/BATCH_SIZE)/barLength;
        std::cout<<"EPOCH: "<<e+1<<" <<<<<<<<<"<<std::endl;
        std::cout << "[";
        Matrix input;
        Matrix label;
        for (int i=0; i<1934; i++){
            if (i == 0){
                input = Matrix(mat_train_data.GetRow(i), 1, 1024);
                label = Matrix(mat_train_label.GetRow(i), 1, 10);
            }
            else if (i % BATCH_SIZE == 0){
                float loss = model.Backpropagation(input, label, criterion, LR);
                outFile << loss <<std::endl;
                if (i % pos == 0){
                    std::cout << "#";
                    std::cout.flush();
                } 
                input = Matrix::Transpose(Matrix(mat_train_data.GetRow(i)));
                label = Matrix::Transpose(Matrix(mat_train_label.GetRow(i)));
                // break;
            }
            else{
                input.AddRow(mat_train_data.GetRow(i));
                label.AddRow(mat_train_label.GetRow(i));
            }
        }
        std::cout << "]\n";
    }
    outFile.close();
    model.SaveMode("model.bin");

    // Matrix output = model.Feedforward(input);
    // Matrix target(10, 1, 1);
    // float loss = model.Backpropagation(input, target, criterion, LR);
    // print(loss);

    return 0;
}