#include <list>
#include <ctime>

#include "mex.h"
#include "matrix.h"

/*
    all matrices must has the same row dimension
    [value, time] = queryFullSearch(A,B,C,kNN);
    A:R x queries,
    B:R x Lb,
    C:R x Lc,
    value:knn x queries
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   
    clock_t start, finish;
    const size_t rank_size = mxGetM(prhs[0]);
    const size_t NumQueries = mxGetN(prhs[0]);
    const size_t knn = mxGetPr(prhs[3])[0];
    // query matrix
    Matrix A(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
    // factor matrices
    Matrix B(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
    Matrix C(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));
    const size_t NumMat = 2;
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
    size_t *maxIdx = (size_t*)malloc(NumMat * sizeof(size_t));
    maxIdx[0] = mxGetN(prhs[1]);
    maxIdx[1] = mxGetN(prhs[2]);
    SubIndex index(NumMat, maxIdx);
    for (int i = 0; i < A.col; ++i){
        std::list<double> listTop;
        mexPrintf(">> Dealing with the %d-th query...\n",i);
        index.reset();
        for(size_t count = 0; count < knn && !index.isDone(); ++index){
            double temp = MatrixColMul(A,B,C,i,index.getIdx()[0],index.getIdx()[1]);
            listTop.push_back(temp);
            ++count;
        }
        listTop.sort();
        listTop.reverse();
        while(!index.isDone()){
            double temp = MatrixColMul(A,B,C,i,index.getIdx()[0],index.getIdx()[1]);
            if(temp > listTop.back()){
                doInsert(temp, listTop);
            }
            ++index;
        } 
        std::list<double>::iterator itr = listTop.begin();
        for(int p = 0; p < knn && p < listTop.size(); ++p){
            knnValue[i*knn + p] = *itr++;
        }
    }
    finish = clock();
    duration[0] = (double)(finish-start)/(NumQueries*CLOCKS_PER_SEC);
    free(maxIdx);
}
