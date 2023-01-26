//
// Created by marti on 18.11.2022.
//

#ifndef MATMUL_MATMUL_SIMPLE_H
#define MATMUL_MATMUL_SIMPLE_H

#include <iostream>
#include "MatMul.h"

using namespace std;

/* ============ Sequential operations ================= */

#define MAT_MUL_LOW_LIMIT_SEQ MUL_LOW_LIMIT_SEQ //256
int MUL_LOW_LIMIT_SEQ = 256;

void transposeMat(Mat_t mat){
    MAT_DATA_T tmp1 = 0;
    MAT_DATA_T tmp2 = 0;
    for (int i = 1; i < mat.size; i++){
        for (int j = 0; j < i; j++){
            tmp1 = mat.getValue(i,j);
            tmp2 = mat.getValue(j,i);
            mat.setValue(tmp2,i,j);
            mat.setValue(tmp1, j, i);
        }
    }
}

void matMulSimple(Mat_t matA, Mat_t matB, Mat_t matC){
//    auto start = std::chrono::high_resolution_clock::now();
    transposeMat(matB);

    MAT_DATA_T sum = 0;
    for (int i = 0; i < matC.size; i++){
        for (int j = 0; j < matC.size; j++){
            sum = 0;
            for (int k = 0; k < matC.size; k++){
                sum += matA.getValue(i,k) * matB.getValue(j, k);
            }
            matC.setValue(sum, i, j);
        }
    }

    transposeMat(matB);
//    auto end = std::chrono::high_resolution_clock::now();
//    auto execTimeSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//    cout << "Mult - " << execTimeSec.count() << endl;
}

void matAdd(Mat_t matA, Mat_t matB, Mat_t matC){
//    auto start = std::chrono::high_resolution_clock::now();
    MAT_DATA_T tmp = 0;
    for (int i = 0; i < matC.size; i++){
        for (int j = 0; j < matC.size; j++){
            tmp = matA.getValue(i, j) + matB.getValue(i, j);
            matC.setValue(tmp,i, j);
        }
    }
//    auto end = std::chrono::high_resolution_clock::now();
//    auto execTimeSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//    cout << "Add - " << execTimeSec.count() << endl;
}

void matSub(Mat_t matA, Mat_t matB, Mat_t matC){
//    auto start = std::chrono::high_resolution_clock::now();
    MAT_DATA_T tmp = 0;
    for (int i = 0; i < matC.size; i++){
        for (int j = 0; j < matC.size; j++){
            tmp = matA.getValue(i, j) - matB.getValue(i, j);
            matC.setValue(tmp,i, j);
        }
    }
//    auto end = std::chrono::high_resolution_clock::now();
//    auto execTimeSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//    cout << "Sub - " << execTimeSec.count() << endl;
}

void matMulStrassen(Mat_t matA, Mat_t matB, Mat_t matC){

    // If matrix smaller than constant then use mult simple
    if (matC.size <= MAT_MUL_LOW_LIMIT_SEQ){
        matMulSimple(matA, matB, matC);
        return;
    }

    int sizeDiv = matC.size / 2;

    // Get recursion depth for current rec call to calculate position in MatM arrays
    int matBufStartPos = getMatBufStartPos(sizeDiv);

    // Submatrixes A
    Mat_t matA11, matA12, matA21, matA22;
    subMatOf(matA, &matA11, &matA12, &matA21, &matA22, sizeDiv);

    // Submatrixes B
    Mat_t matB11, matB12, matB21, matB22;
    subMatOf(matB, &matB11, &matB12, &matB21, &matB22, sizeDiv);

    // Initialize M 1-7
    Mat_t matM[7];
    for (int i = 0; i < 7; i++){
        matM[i].row_size = sizeDiv;
        matM[i].size = sizeDiv;
        matM[i].mat = matM_buf + matBufStartPos + i*sizeDiv*sizeDiv;
    }

    Mat_t matTmp[10]; // Temporary results of add/sub
    for (int i = 0; i < 10; i++){
        matTmp[i].row_size = sizeDiv;
        matTmp[i].size = sizeDiv;
        matTmp[i].mat = matM_buf + matBufStartPos + (i+7)*sizeDiv*sizeDiv;
    }

    // M1 = (A11 + A22)(B11 + B22)
    matAdd(matA11, matA22, matTmp[0]);
    matAdd(matB11, matB22, matTmp[1]);
    matMulStrassen(matTmp[0], matTmp[1], matM[0]);

    // M2 = (A21 + A22)(B11)
    matAdd(matA21, matA22, matTmp[2]);
    matMulStrassen(matTmp[2], matB11, matM[1]);

    // M3 = (A11)(B12 - B22)
    matSub(matB12, matB22, matTmp[3]);
    matMulStrassen(matA11, matTmp[3], matM[2]);

    // M4 = (A22)(B21 - B11)
    matSub(matB21, matB11, matTmp[4]);
    matMulStrassen(matA22, matTmp[4], matM[3]);

    // M5 = (A11 + A12)(B22)
    matAdd(matA11, matA12, matTmp[5]);
    matMulStrassen(matTmp[5], matB22, matM[4]);

    // M6 = (A21 - A11)(B11 + B12)
    matSub(matA21, matA11, matTmp[6]);
    matAdd(matB11, matB12, matTmp[7]);
    matMulStrassen(matTmp[6], matTmp[7], matM[5]);

    // M7 = (A12 - A22)(B21 + B22)
    matSub(matA12, matA22, matTmp[8]);
    matAdd(matB21, matB22, matTmp[9]);
    matMulStrassen(matTmp[8], matTmp[9], matM[6]);

    // Submatrixes C
    Mat_t matC11, matC12, matC21, matC22;
    subMatOf(matC, &matC11, &matC12, &matC21, &matC22, sizeDiv);

    // C11 = M1 + M4 - M5 +M7
    matAdd(matM[0], matM[3], matC11);
    matSub(matC11, matM[4], matC11);
    matAdd(matC11, matM[6], matC11);

    // C12 = M3 + M5
    matAdd(matM[2], matM[4], matC12);

    // C21 = M2 + M4
    matAdd(matM[1], matM[3], matC21);

    // C22 = M1 - M2 + M3 + M6
    matSub(matM[0], matM[1], matC22);
    matAdd(matC22, matM[2], matC22);
    matAdd(matC22, matM[5], matC22);
}


#endif //MATMUL_MATMUL_SIMPLE_H
