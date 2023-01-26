# Makefile for Writing Make Files Example
 
# *****************************************************
# Variables to control Makefile operation
 
CC = g++
CFLAGS = -Wall -std=c++17 -O3 -fopenmp -ftree-vectorize -mavx -funroll-loops -ffast-math -fopt-info-vec
 
# ****************************************************
# Targets needed to bring the executable up to date

all: main

main: main.o
	$(CC) $(CFLAGS) -o matmul main.o 
 
# The main.o target can be written more simply
 
main.o: main.cpp MatMul_simple.h MatMul_parallel.h MatMul_parDep.h MatMul_cache.h MatMul.h
	$(CC) $(CFLAGS) -c main.cpp
