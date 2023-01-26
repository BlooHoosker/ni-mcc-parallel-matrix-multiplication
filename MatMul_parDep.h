//
// Created by marti on 18.11.2022.
//

#ifndef MATMUL_MATMUL_PARDEP_H
#define MATMUL_MATMUL_PARDEP_H

#include <iostream>
#include "MatMul.h"
#include "MatMul_simple.h"
#include "MatMul_cache.h"

using namespace std;

/* ============ Paralel with dependencies ================= */

#define MAT_MUL_LOW_LIMIT_PAR_DEP MUL_LOW_LIMIT_PAR_DEP // 512
int MUL_LOW_LIMIT_PAR_DEP = 512;

void matMulStrassenParDep(Mat_t matA, Mat_t matB, Mat_t matC){

    // If matrix smaller than constant then use mult simple
    if (matC.size <= MAT_MUL_LOW_LIMIT_PAR_DEP){
        matMulSimpleCache(matA, matB, matC);
        return;
    }

    int sizeDiv = matC.size / 2;

    // Submatrixes A
    Mat_t matA11, matA12, matA21, matA22;
    subMatOf(matA, &matA11, &matA12, &matA21, &matA22, sizeDiv);

    // Submatrixes B
    Mat_t matB11, matB12, matB21, matB22;
    subMatOf(matB, &matB11, &matB12, &matB21, &matB22, sizeDiv);

    // Submatrixes C
    Mat_t matC11, matC12, matC21, matC22;
    subMatOf(matC, &matC11, &matC12, &matC21, &matC22, sizeDiv);

    // Initialize M 1-7
    Mat_t matM[7];
    for (int i = 0; i < 7; i++){
        allocateMat(&(matM[i]), sizeDiv, true);
    }

    Mat_t matTmp[14]; // Temporary results of add/sub
    for (int i = 0; i < 14; i++){
        allocateMat(&(matTmp[i]), sizeDiv, true);
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            // M1 = (A11 + A22)(B11 + B22)
            #pragma omp task depend(out:matTmp[0])
            matAdd(matA11, matA22, matTmp[0]);

            #pragma omp task depend(out:matTmp[1])
            matAdd(matB11, matB22, matTmp[1]);

            #pragma omp task depend(in:matTmp[0],matTmp[1]) depend(out:matM[0])
            matMulStrassenParDep(matTmp[0], matTmp[1], matM[0]);


            // M2 = (A21 + A22)(B11)
            #pragma omp task depend(out:matTmp[2])
            matAdd(matA21, matA22, matTmp[2]);

            #pragma omp task depend(in:matTmp[2]) depend(out:matM[1])
            matMulStrassenParDep(matTmp[2], matB11, matM[1]);


            // M3 = (A11)(B12 - B22)
            #pragma omp task depend(out:matTmp[3])
            matSub(matB12, matB22, matTmp[3]);

            #pragma omp task depend(in:matTmp[3]) depend(out:matM[2])
            matMulStrassenParDep(matA11, matTmp[3], matM[2]);


            // M4 = (A22)(B21 - B11)
            #pragma omp task depend(out:matTmp[4])
            matSub(matB21, matB11, matTmp[4]);

            #pragma omp task depend(in:matTmp[4]) depend(out:matM[3])
            matMulStrassenParDep(matA22, matTmp[4], matM[3]);


            // M5 = (A11 + A12)(B22)
            #pragma omp task depend(out:matTmp[5])
            matAdd(matA11, matA12, matTmp[5]);

            #pragma omp task depend(in:matTmp[5]) depend(out:matM[4])
            matMulStrassenParDep(matTmp[5], matB22, matM[4]);


            // M6 = (A21 - A11)(B11 + B12)
            #pragma omp task depend(out:matTmp[6])
            matSub(matA21, matA11, matTmp[6]);

            #pragma omp task depend(out:matTmp[7])
            matAdd(matB11, matB12, matTmp[7]);

            #pragma omp task depend(in:matTmp[6],matTmp[7]) depend(out:matM[5])
            matMulStrassenParDep(matTmp[6], matTmp[7], matM[5]);


            // M7 = (A12 - A22)(B21 + B22)
            #pragma omp task depend(out:matTmp[8])
            matSub(matA12, matA22, matTmp[8]);

            #pragma omp task depend(out:matTmp[9])
            matAdd(matB21, matB22, matTmp[9]);

            #pragma omp task depend(in:matTmp[8],matTmp[9]) depend(out:matM[6])
            matMulStrassenParDep(matTmp[8], matTmp[9], matM[6]);


            // C11 = M1 + M4 - M5 + M7
            #pragma omp task depend(in:matM[0], matM[3]) depend(out:matTmp[10])
            matAdd(matM[0], matM[3], matTmp[10]);

            #pragma omp task depend(in:matM[4], matM[6]) depend(out:matTmp[11])
            matSub(matM[6], matM[4], matTmp[11]);

            #pragma omp task depend(in:matTmp[10], matTmp[11])
            matAdd(matTmp[10], matTmp[11], matC11);


            // C12 = M3 + M5
            #pragma omp task depend(in:matM[2], matM[4])
            matAdd(matM[2], matM[4], matC12);


            // C21 = M2 + M4
            #pragma omp task depend(in:matM[1], matM[3])
            matAdd(matM[1], matM[3], matC21);


            // C22 = M1 - M2 + M3 + M6
            #pragma omp task depend(in:matM[0], matM[1]) depend(out:matTmp[12])
            matSub(matM[0], matM[1], matTmp[12]);

            #pragma omp task depend(in:matM[2], matM[5]) depend(out:matTmp[13])
            matAdd(matM[2], matM[5], matTmp[13]);

            #pragma omp task depend(in:matTmp[12], matTmp[13])
            matAdd(matTmp[12], matTmp[13], matC22);
        }
    }

    for (int i = 0; i < 7; i++){
        freeMat(matM[i]);
    }
    for (int i = 0; i < 14; i++){
        freeMat(matTmp[i]);
    }

}

#endif //MATMUL_MATMUL_PARDEP_H
