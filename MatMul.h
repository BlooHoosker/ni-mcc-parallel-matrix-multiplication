//
// Created by marti on 18.11.2022.
//

#ifndef MATMUL_MATMUL_H
#define MATMUL_MATMUL_H

#include <iostream>

typedef float MAT_DATA_T;

int topMatMSize;
MAT_DATA_T * matM_buf;

typedef struct {
    MAT_DATA_T * __restrict__ mat; // 1D Buffer for matrix values
    int size; // Size of current matrix
    int row_size; // Size of TOP row matrix, in case this matrix is a submatrix of other matrix
    MAT_DATA_T getValue(int row, int col) const{ return mat[row*row_size+col]; }
    void setValue(MAT_DATA_T value, int row, int col){ mat[row*row_size+col] = value; }
} Mat_t;

void allocateMat(Mat_t * mat, int size, bool init = false){

    mat->row_size = size;
    mat->size = size;

    if (init){
        // Initializing matrix C with zeros
        (*mat).mat = new MAT_DATA_T [size*size]();
    } else {
        (*mat).mat = new MAT_DATA_T[size*size];
    }
}

void freeMat(Mat_t mat){
    delete [] mat.mat;
}

void printMat(Mat_t mat){
    for (int i = 0; i < mat.size; i++){
        for (int j = 0; j < mat.size; j++){
            std::cout << mat.getValue(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void randomizeMat(Mat_t mat){
    for (int i = 0; i < mat.size; i++){
        for (int j = 0; j < mat.size; j++){
            mat.setValue(rand() % 10, i, j);
        }
    }
}

void subMatOf(Mat_t mat, Mat_t * sub11, Mat_t * sub12, Mat_t * sub21, Mat_t * sub22, int subSize){
    *sub11 = mat;
    (*sub11).size = subSize;

    *sub12 = mat;
    (*sub12).mat += subSize;
    (*sub12).size = subSize;

    *sub21 = mat;
    (*sub21).mat += subSize*mat.row_size;
    (*sub21).size = subSize;

    *sub22 = mat;
    (*sub22).mat += subSize*mat.row_size + subSize;
    (*sub22).size = subSize;
}

int getMatBufStartPos(int currMatSize){
    int tmp = topMatMSize;
    int pos = 0;
    while(tmp != currMatSize) {
        pos += (tmp*tmp)*21;
        tmp /= 2;
    }
    return pos;
}

void allocateMBuf(int matSize){
    matM_buf = new MAT_DATA_T [2*21*((matSize/2)*(matSize/2))]();
}

void freeMBuf(){
    delete [] matM_buf;
}

bool isEqual(MAT_DATA_T a, MAT_DATA_T b)
{
    return std::fabs(a - b) <= (std::numeric_limits<float>::epsilon());
    //return std::fabs(((int) a - (int) b)) <= 1;
}

void compareMat(Mat_t matA, Mat_t matB){
    if (matA.size != matB.size){
        std::cout << "Matrix sizes don't match!" << std::endl;
    }

    for (int i = 0; i < matB.size; i++){
        for (int j = 0; j < matB.size; j++){
            if (!isEqual(matA.getValue(i,j), matB.getValue(i,j))){
                std::cout << "Matrix values don't match!" << std::endl;
                std::cout << matA.getValue(i,j) << " " << matB.getValue(i,j) << std::endl;
                return;
            }
        }
    }
}

#endif //MATMUL_MATMUL_H
