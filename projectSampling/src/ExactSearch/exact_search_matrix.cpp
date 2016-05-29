#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include "mex.h"

#define TOP1K 1000

typedef struct 
{
    size_t row;
    size_t col;
    double *element;
}Matrix;

int doInsert(double value, double* toInsert, int num){
    double front,next;
    for(int i = 0; i < num; ++i){
        if(value > toInsert[i]){
            // find ans insert
            front = toInsert[i];
            toInsert[i] = value;
            // shift the left element
            for(int j = i + 1; j < num; ++j){
                next = toInsert[j];
                toInsert[j] = front;
                front = next;
            }
            return i;
        }
    }
    return num;
}



/*
    all matrix has the same row, which is the dimension of feature
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   
    clock_t start, finish;
    double duration;
    //-----------------
    // Initialization 
    //-----------------
    start = clock();
    Matrix MatA,MatB;
    MatA.element = mxGetPr(prhs[0]);
    MatA.row = mxGetM(prhs[0]);
    MatA.col = mxGetN(prhs[0]);
    MatB.element = mxGetPr(prhs[1]);
    MatA.row = mxGetM(prhs[1]);
    MatB.col = mxGetN(prhs[1]);
    finish = clock();
    duration = (double)(finish-start) / CLOCKS_PER_SEC;
    printf("%f seconds during initialization\n", duration);
    //----------------------------
    //  Compute the top 1K values
    //----------------------------
    start = clock();
    int dim = MatA.row;
    plhs[0] = mxCreateDoubleMatrix(TOP1K, 1, mxREAL);
    double *top = mxGetPr(plhs[0]);
    double temp = 0;
    for(int k = 0; k < dim; ++k){
        temp += MatA.element[k]*MatB.element[k];
    }    
    for (int i = 0; i < TOP1K; ++i){
        top[i] = temp;
    }
    for(size_t i = 0; i < MatA.col; ++i){
        for(size_t j = 0; j < MatB.col; ++j){
            temp = 0.0;
            for(int k = 0; k < dim; ++k){
                temp += MatA.element[i*dim+k]*MatB.element[j*dim+k];
            }
            if(temp > top[TOP1K - 1]){
                doInsert(temp, top, TOP1K);
            }
        }
    }
    finish = clock();
    duration = (double)(finish-start) / CLOCKS_PER_SEC;
    printf("%f seconds during computing\n", duration);
}