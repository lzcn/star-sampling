#include <list>
#include <ctime>
#include "mex.h"
#include "../../include/matrix.h"

/*
    all matrices must has the same row dimension
    [value,time] = queryFullSearch(A,B,C,kNN);
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
    size_t *max = (size_t*)malloc(2*sizeof(size_t));
    max[0] = mxGetN(prhs[1]);
    max[1] = mxGetN(prhs[2]);
    SubIndex index(2,max);
    for (int i = 0; i < A.col; ++i){
        std::list<double> listTop;
        mexPrintf("Dealing with the %d-th query...\n",i);
        index.reset();
        for(size_t count = 0; count < knn && !index.isDone(); ++index){
            temp = MatrixColMul(A,B,C,i,index.getIdx()[0],index.getIdx()[1]);
            listTop.push_back(temp);
            ++count;
        }
        listTop.sort();
        listTop.reverse();
        while(!index.isDone()){
            temp = MatrixColMul(A,B,C,i,index.getIdx()[0],index.getIdx()[1]);
            if(temp > listTop.back()){
                doInsert(temp, listTop);
            }
            ++index;
        } 
        std::list<double>::iterator itr = listTop.begin();
        for(int p = 0; p < knn && p < listTop.size(); ++p){
            knnValue[i*knn + p] = (*itr);
            ++itr;
        }
    }
    finish = clock();
    duration[0] = (double)(finish-start)/(NumQueries*CLOCKS_PER_SEC);
    free(max);
}