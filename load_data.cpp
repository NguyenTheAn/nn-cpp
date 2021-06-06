#include <iostream>
#include <fstream>
#include "lib/matrix/matrix.cpp"
#include "lib/activationfunction/activation.cpp"
#include "lib/lossfunction/lossfunction.cpp"
#include "lib/layer/layer.cpp"
#include "stdio.h"
#include "string.h"

#define print(x) std::cout << x << std::endl


Matrix load_data(const char* data_path, int size){
    freopen (data_path,"r",stdin);
    std::vector<double> data(0);
    for (int i = 0; i < size; i++){
        int pixel;
        std::cin >> pixel;
        data.push_back(pixel);
    }
    Matrix mat = Matrix(data);

    return mat;
}

int main(){
    Matrix mat_train_data = load_data("data/train_data.txt", 1934 * 32 * 32);
    Matrix mat_train_label = load_data("data/train_label.txt", 1934);
    Matrix mat_valid_data = load_data("data/valid_data.txt", 500 * 32 * 32);
    Matrix mat_valid_label = load_data("data/valid_label.txt", 500);
    Matrix mat_test_data = load_data("data/test_data.txt", 446 * 32 * 32);
    Matrix mat_test_label = load_data("data/test_label.txt", 446);
}