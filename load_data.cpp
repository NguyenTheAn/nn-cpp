#include <iostream>
#include <fstream>
#include "lib/matrix/matrix.cpp"
#include "lib/activationfunction/activation.cpp"
#include "lib/lossfunction/lossfunction.cpp"
#include "lib/layer/layer.cpp"
#include "stdio.h"
#include "string.h"

#define print(x) std::cout << x << std::endl


Matrix load_data(const char* data_path, int rows, int cols){
    freopen (data_path,"r",stdin);
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(0); std::cout.tie(0);
    std::vector<double> data(0);
    for (int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            int pixel;
            std::cin >> pixel;
            data.push_back(pixel);
            // print(pixel);
        }
    }
    Matrix mat = Matrix(data);
    // print(mat);
    mat.m_Rows = rows;
    mat.m_Columns = cols;
    return mat;
}

int main(){

    Matrix mat_valid_data = load_data("/home/long/nn-cpp/data/valid_data.txt", 500, 1024);
    Matrix mat_valid_label = load_data("/home/long/nn-cpp/data/valid_label.txt", 500, 10);

    Matrix mat_train_data = load_data("/home/long/nn-cpp/data/train_data.txt", 1934, 1024);
    Matrix mat_train_label = load_data("/home/long/nn-cpp/data/train_label.txt", 1934, 10);

    Matrix mat_test_data = load_data("/home/long/nn-cpp/data/test_data.txt", 446, 1024);
    Matrix mat_test_label = load_data("/home/long/nn-cpp/data/test_label.txt", 446, 10);

    print(mat_valid_label);
}