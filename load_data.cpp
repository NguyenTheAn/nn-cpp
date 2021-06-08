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
            float pixel;
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


    // optical recognition
    Matrix mat_valid_data = load_data("../data/valid_data.txt", 500, 1024);
    Matrix mat_valid_label = load_data("../data/valid_label.txt", 500, 10);

    Matrix mat_train_data = load_data("../data/train_data.txt", 1934, 1024);
    Matrix mat_train_label = load_data("../data/train_label.txt", 1934, 10);

    Matrix mat_test_data = load_data("../data/test_data.txt", 446, 1024);
    Matrix mat_test_label = load_data("../data/test_label.txt", 446, 10);

    // spam classification
    Matrix mat_valid_data = load_data("../data/spam_valid_data.txt", 920, 57);
    Matrix mat_valid_label = load_data("../data/spam_valid_label.txt", 920, 2);

    Matrix mat_train_data = load_data("../data/spam_train_data.txt", 2760, 57);
    Matrix mat_train_label = load_data("../data/spam_train_label.txt", 2760, 2);

    Matrix mat_test_data = load_data("../data/spam_test_data.txt", 921, 57);
    Matrix mat_test_label = load_data("../data/spam_test_label.txt", 921, 2);

    // letter recognition
    Matrix mat_valid_data = load_data("../data/letter_valid_data.txt", 4000, 16);
    Matrix mat_valid_label = load_data("../data/letter_valid_label.txt", 4000, 26);

    Matrix mat_train_data = load_data("../data/letter_train_data.txt", 12000, 16);
    Matrix mat_train_label = load_data("../data/letter_train_label.txt", 12000, 26);

    Matrix mat_test_data = load_data("../data/letter_test_data.txt", 4000, 16);
    Matrix mat_test_label = load_data("../data/letter_test_label.txt", 4000, 26);
}