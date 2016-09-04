/*
    Exhaustive search for three factor matrices factor matrices.
    [value, time, cordinates] = exactSearchThreeOrderTrensor(A, B, C, top_t)
    Inputs:
        A, B, C: factor matrices, same row dimension.
        top_t : the top_t value to find
    Outputs: 
        value: size (top_t,1)
        time:
        coordinates: size (top_t,3)
    
    Author : Zhi Lu
*/

#include <list>
#include <ctime>

#include "mex.h"
#include "matrix.h"
#include "utilmex.h"
double GetValue(Matrix &A, Matrix &B, Matrix &C, Matrix &D, uint i,uint j, uint k, uint t){
    uint row = A.row;
    double ans = 0.0;
    for(uint r = 0; r < row; ++r){
        ans += A(r,i)*B(r,j)*C(r,k)*D(r,t);
    }
    return ans;
}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   

    clock_t start, finish;
    double *duration;
    Matrix A(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]),MATRIX_NONE_SUM);
    Matrix B(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]),MATRIX_NONE_SUM);
    Matrix C(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]),MATRIX_NONE_SUM);
    Matrix D(mxGetM(prhs[3]),mxGetN(prhs[3]),mxGetPr(prhs[3]),MATRIX_NONE_SUM);
    const int top_t = mxGetPr(prhs[4])[0];
    plhs[0] = mxCreateDoubleMatrix(top_t,1,mxREAL);
    double *topValue = mxGetPr(plhs[0]);
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    duration = mxGetPr(plhs[1]);
    mexPrintf("Start Exhaustive Search...\n");mexEvalString("drawnow");
    double total = mxGetN(prhs[0])*mxGetN(prhs[1])*mxGetN(prhs[2])*mxGetN(prhs[3]);
    double progress = 0;
    double flag = 0;
    std::list<double> listTop;
    for (size_t i = 0; i < top_t; i++) {
        listTop.push_back(-1000);
    }
    progressbar(0);
    start = clock();
    for (size_t i = 0; i < mxGetN(prhs[0]); i++) {
        for (size_t j = 0; j < mxGetN(prhs[1]); j++) {
            for (size_t k = 0; k < mxGetN(prhs[2]); k++) {
                for (size_t t = 0; t < mxGetN(prhs[3]); t++) {
                    double temp = GetValue(A,B,C,D,i,j,k,t);
                    doInsert(temp, listTop);
                    progress += 1;
                    flag += 1;
                    if(flag > 1e8){
                        clearprogressbar();
                        progressbar(progress/total);
                        flag = 0;
                    }
                }
            }        
        }
    }
    finish = clock();
    duration[0] = (double)(finish - start)/CLOCKS_PER_SEC;
    //---------------------------------
    // convert result to Matlab format
    //---------------------------------
    auto itr = listTop.begin();
    for(uint i = 0; i < top_t; ++i){
        topValue[i] = *itr++;
    }
    clearprogressbar();
    progressbar(1);
    mexPrintf("\n");
}
