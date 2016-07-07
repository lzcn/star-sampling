#include <list>
#include <ctime>
#include "mex.h"
#include "../../include/matrix.h"


/*
    the matrix must has the same row dimension
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   
    clock_t start, finish;
    const int rank_size = mxGetM(prhs[0]);
    const size_t NumQueries = mxGetN(prhs[0]);
    const int knn = mxGetPr(prhs[3])[0];
    // query matrix
    Matrix A(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
    // factor matrices
    Matrix B(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
    Matrix C(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));
    // result values for each query
    plhs[0] = mxCreateDoubleMatrix(knn, NumQueries, mxREAL);
    double *knnValue = mxGetPr(plhs[0]);
    // sampling time for each query
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);  
    double *duration = mxGetPr(plhs[1]);
    mexPrintf("Initialization Complete!\n");
    //---------------------
    // Start full search
    //---------------------
    mexPrintf("Start full search!\n");
    start = clock();
    double temp = 0.0;
    for (int i = 0; i < A.col; ++i){
        std::list<double> listTop;
        int count  = 0;
        int k = 0;
        mexPrintf("Dealing with the %d-th query...\n",i);
        while(count < knn &&  k < C.col){
            int j = 0;
            while(count < knn && j < B.col){
                temp = MatrixColMul(A,B,C,i,j,k);
                listTop.push_back(temp);
                count++;
                ++j;    
            }
            ++k;
        }
        listTop.sort(); 
        listTop.reverse();
        for(int j = 0; j < B.col; ++j){
            for(int k = 0; k < C.col; ++k){
                temp = MatrixColMul(A,B,C,i,j,k);
                if(temp > listTop.back()){
                    doInsert(temp, listTop);
                }
            }
        }
        std::list<double>::iterator itr = listTop.begin();
        for(int p = 0; p < knn && p < listTop.size(); ++p){
            knnValue[i*knn + p] = (*itr);
            ++itr;
        }
    }
    finish = clock();
    duration[0] = (double)(finish-start)/CLOCKS_PER_SEC;
}