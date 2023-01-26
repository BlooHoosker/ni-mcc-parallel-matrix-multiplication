//
// Created by marti on 18.11.2022.
//

#ifndef MATMUL_MATMUL_CACHE_H
#define MATMUL_MATMUL_CACHE_H

#include <iostream>
#include "MatMul.h"

/* ============ Sequential operations cache optimized ================= */

#define MAT_MUL_LOW_LIMIT_CACHE MUL_LOW_LIMIT_CACHE //512
int MUL_LOW_LIMIT_CACHE = 512;

#define CACHE_LS 64
#define INNER_STEP 2
void matMulSimpleCache(Mat_t matA, Mat_t matB, Mat_t matC){

    MAT_DATA_T sum1, sum2, sum3, sum4;
    int n = matC.size;

    int n2 = matC.size*matC.size;
    for (int i = 0; i < n2; i++){
        matC.mat[i] = 0;
    }

    alignas(32) MAT_DATA_T matA_tmp[CACHE_LS*CACHE_LS];
    alignas(32) MAT_DATA_T matB_tmp[CACHE_LS*CACHE_LS];

    for (int i = 0; i < n; i += CACHE_LS) {
        for (int k = 0; k < n; k += CACHE_LS) {

            // Copy to A
            for (int x = 0; x < CACHE_LS; x += INNER_STEP) {
                for (int y = 0; y < CACHE_LS; y += INNER_STEP) {
                    matA_tmp[x * CACHE_LS + y] = matA.getValue((x + i), y + k);
                    matA_tmp[(x+1) * CACHE_LS + y] = matA.getValue((x+1 + i), y + k);
                    matA_tmp[x * CACHE_LS + y+1] = matA.getValue((x + i), y+1 + k);
                    matA_tmp[(x+1) * CACHE_LS + y+1] = matA.getValue((x+1 + i), y+1 + k);
                }
            }

            for (int j = 0; j < n; j += CACHE_LS) {

                // Copy to B transposed
                for (int x = 0; x < CACHE_LS; x += INNER_STEP) {
                    for (int y = 0; y < CACHE_LS; y += INNER_STEP) {
                        matB_tmp[x * CACHE_LS + y] = matB.getValue((y + k), x + j);
                        matB_tmp[(x+1) * CACHE_LS + y] = matB.getValue((y + k), x+1 + j);
                        matB_tmp[x * CACHE_LS + y+1] = matB.getValue((y+1 + k), x + j);
                        matB_tmp[(x+1) * CACHE_LS + y+1] = matB.getValue((y+1 + k), x+1 + j);
                    }
                }

                for (int i2 = 0; i2 < CACHE_LS; i2 += INNER_STEP) {
                    for (int j2 = 0; j2 < CACHE_LS; j2 += INNER_STEP) {
                        sum1 = sum2 = sum3 = sum4 = 0;
                        for (int k2 = 0; k2 < CACHE_LS; k2++) {
                            sum1 += matA_tmp[(i2) * CACHE_LS + k2] * matB_tmp[(j2) * CACHE_LS + k2];
                            sum2 += matA_tmp[(i2) * CACHE_LS + k2] * matB_tmp[(j2+1) * CACHE_LS + k2];
                            sum3 += matA_tmp[(i2+1) * CACHE_LS + k2] * matB_tmp[(j2) * CACHE_LS + k2];
                            sum4 += matA_tmp[(i2+1) * CACHE_LS + k2] * matB_tmp[(j2+1) * CACHE_LS + k2];
                        }

                        sum1 += matC.getValue((i2 + i),(j2 + j));
                        matC.setValue(sum1, (i2 + i),(j2 + j));

                        sum2 += matC.getValue((i2 + i),(j2+1 + j));
                        matC.setValue(sum2, (i2 + i),(j2+1 + j));

                        sum3 += matC.getValue((i2+1 + i),(j2 + j));
                        matC.setValue(sum3, (i2+1 + i),(j2 + j));

                        sum4 += matC.getValue((i2+1 + i),(j2+1 + j));
                        matC.setValue(sum4, (i2+1 + i),(j2+1 + j));
                    }
                }
            }
        }
    }

}

void matAddCache(Mat_t matA, Mat_t matB, Mat_t matC){
    MAT_DATA_T tmp = 0;
    for (int i = 0; i < matC.size; i++){
        for (int j = 0; j < matC.size; j++){
            tmp = matA.getValue(i, j) + matB.getValue(i, j);
            matC.setValue(tmp,i, j);
        }
    }
}

void matSubCache(Mat_t matA, Mat_t matB, Mat_t matC){
    MAT_DATA_T tmp = 0;
    for (int i = 0; i < matC.size; i++){
        for (int j = 0; j < matC.size; j++){
            tmp = matA.getValue(i, j) - matB.getValue(i, j);
            matC.setValue(tmp,i, j);
        }
    }
}

void matMulStrassenCache(Mat_t matA, Mat_t matB, Mat_t matC){

    // If matrix smaller than constant then use mult simple
    if (matC.size <= MAT_MUL_LOW_LIMIT_CACHE){
        matMulSimpleCache(matA, matB, matC);
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
    matAddCache(matA11, matA22, matTmp[0]);
    matAddCache(matB11, matB22, matTmp[1]);
    matMulStrassenCache(matTmp[0], matTmp[1], matM[0]);

    // M2 = (A21 + A22)(B11)
    matAddCache(matA21, matA22, matTmp[2]);
    matMulStrassenCache(matTmp[2], matB11, matM[1]);

    // M3 = (A11)(B12 - B22)
    matSubCache(matB12, matB22, matTmp[3]);
    matMulStrassenCache(matA11, matTmp[3], matM[2]);

    // M4 = (A22)(B21 - B11)
    matSubCache(matB21, matB11, matTmp[4]);
    matMulStrassenCache(matA22, matTmp[4], matM[3]);

    // M5 = (A11 + A12)(B22)
    matAddCache(matA11, matA12, matTmp[5]);
    matMulStrassenCache(matTmp[5], matB22, matM[4]);

    // M6 = (A21 - A11)(B11 + B12)
    matSubCache(matA21, matA11, matTmp[6]);
    matAddCache(matB11, matB12, matTmp[7]);
    matMulStrassenCache(matTmp[6], matTmp[7], matM[5]);

    // M7 = (A12 - A22)(B21 + B22)
    matSubCache(matA12, matA22, matTmp[8]);
    matAddCache(matB21, matB22, matTmp[9]);
    matMulStrassenCache(matTmp[8], matTmp[9], matM[6]);

    // Submatrixes C
    Mat_t matC11, matC12, matC21, matC22;
    subMatOf(matC, &matC11, &matC12, &matC21, &matC22, sizeDiv);

    // C11 = M1 + M4 - M5 +M7
    matAddCache(matM[0], matM[3], matC11);
    matSubCache(matC11, matM[4], matC11);
    matAddCache(matC11, matM[6], matC11);

    // C12 = M3 + M5
    matAddCache(matM[2], matM[4], matC12);

    // C21 = M2 + M4
    matAddCache(matM[1], matM[3], matC21);

    // C22 = M1 - M2 + M3 + M6
    matSubCache(matM[0], matM[1], matC22);
    matAddCache(matC22, matM[2], matC22);
    matAddCache(matC22, matM[5], matC22);

}



#endif //MATMUL_MATMUL_CACHE_H
