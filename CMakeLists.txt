cmake_minimum_required(VERSION 3.16)
project(MatMul)

set(CMAKE_CXX_STANDARD 17)

add_executable(MatMul main.cpp MatMul.h  MatMul_simple.h MatMul_parallel.h MatMul_parallel.h MatMul_parDep.h MatMul_cache.h)
