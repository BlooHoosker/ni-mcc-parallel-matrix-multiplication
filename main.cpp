#include <iostream>
#include <omp.h>
#include <chrono>
#include <cmath>
#include "MatMul_simple.h"
#include "MatMul_parallel.h"
#include "MatMul_parDep.h"
#include "MatMul_cache.h"

using namespace std;

// todo Use float but pay attention to rounding error

#define MAT_SIZE MAT_SIZE_VAR //2048
int MAT_SIZE_VAR = 2048;

int main(int argc, char** argv) {

    // Arguments: mat_size, strassen_limit, algo sel, core count
    if (argc < 4) {
        cout << "Not enough Arguments" << endl;
        return 1;
    }

    MAT_SIZE_VAR = stoi(string(argv[1]));
    MUL_LOW_LIMIT_SEQ = MUL_LOW_LIMIT_CACHE = MUL_LOW_LIMIT_PAR = MUL_LOW_LIMIT_PAR_DEP = stoi(string(argv[2]));

    int thread_num = stoi(string(argv[4]));
    omp_set_num_threads(thread_num);

    //cout << "=================================================" << endl;
    cout << "THREAD NUM: " << thread_num << endl;
    cout << "MATRIX SIZE: " << MAT_SIZE_VAR << endl;
    cout << "STRASS LIMIT: " << MUL_LOW_LIMIT_SEQ << endl;

    // Initialize random num generator
    int seed = 42;
    srand(seed);

    Mat_t matA, matB, matC, matRef;

    topMatMSize = MAT_SIZE/2;

    // Allocate matrixes A B C
    //std::cout << "Allocation" << std::endl;
    allocateMat(&matA, MAT_SIZE);
    allocateMat(&matB, MAT_SIZE);

    //std::cout << "Randomization" << std::endl;
    randomizeMat(matA);
    randomizeMat(matB);

//    cout << endl;
//    printMat(matA);
//    cout << endl;
//    printMat(matB);
//    cout << endl;

//    std::cout << "Add" << std::endl;
//    matAdd(matA, matB, matC);
//    printMat(matC);
//
//    cout << endl;
//
//    std::cout << "Sub" << std::endl;
//    matSub(matA, matB, matC);
//    printMat(matC);
//
//    cout << endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto execTimeSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    /* ========================== SIMPLE MULT ========================== */
//    allocateMat(&matRef, MAT_SIZE, true);
//
//    std::cout << "Mult - ";
//    start = std::chrono::high_resolution_clock::now();
//    matMulSimple(matA, matB, matRef);
//    end = std::chrono::high_resolution_clock::now();
//    execTimeSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//    cout << "Execution time: " << execTimeSec.count() << " ms" << endl;

//    allocateMBuf(MAT_SIZE);
//    allocateMat(&matC, MAT_SIZE, true);
//
//    std::cout << "Mult Cache - ";
//    start = std::chrono::high_resolution_clock::now();
//    matMulSimpleCache(matA, matB, matC);
//    end = std::chrono::high_resolution_clock::now();
//    execTimeSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//    cout << "Execution time: " << execTimeSec.count() << " ms" << endl;
////    printMat(matRef);
//
//    freeMBuf();
//    freeMat(matC);

    //omp_set_num_threads(12);

    int i = stoi(string(argv[3]));
    //for (int i = 0; i < 5; i++){

        // Allocate buffer for M matrixes for all recursions
        // 2x size of all biggest M+tmp matrixes
        allocateMBuf(MAT_SIZE);
        allocateMat(&matC, MAT_SIZE, true);

        start = std::chrono::high_resolution_clock::now();
        switch(i) {
            case 0:
                matMulStrassen(matA, matB, matC);
                cout << "Strass - ";
                break;
            case 1:
                matMulStrassenCache(matA, matB, matC);
                cout << "Strass Cache Opt - ";
                break;
            case 2:
                matMulStrassenPar(matA, matB, matC);
                cout << "Strass Par - ";
                break;
            case 3:
                matMulStrassenParCache(matA, matB, matC);
                cout << "Strass Par Cache Opt- ";
                break;
            case 4:
                matMulStrassenParDep(matA, matB, matC);
                cout << "Strass Par Dep - ";
                break;
            default:
                matMulSimple(matA, matB, matC);
                cout << "Mult Simple - ";

        }
        end = std::chrono::high_resolution_clock::now();
        execTimeSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        cout << "Execution time: " << execTimeSec.count() << " ms" << endl;
        //compareMat(matRef, matC);
        //printMat(matC);

        freeMBuf();
        freeMat(matC);
    //}

    freeMat(matA);
    freeMat(matB);
    freeMat(matRef);

    return 0;
}
