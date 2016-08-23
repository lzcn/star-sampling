#include <list>
#include <ctime>

#include "mex.h"
#include "matrix.h"
#include "utilmex.h"

/*
    all matrices must has the same row dimension
    [value, time] = queryFullSearch(A,B,C,kNN);
    A:R x number of queries,
    B:R x Lb,
    C:R x Lc,
    value:knn x number of queries
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   
    clock_t start, finish;
    const size_t NumQueries = mxGetN(prhs[0]);
    // factor matrices
    Matrix A(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
    Matrix B(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
    Matrix C(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));
    const size_t knn = mxGetPr(prhs[3])[0];
    const size_t NumMat = 2;
    // result values for each query
    plhs[0] = mxCreateDoubleMatrix(knn, NumQueries, mxREAL);
    double *knnValue = mxGetPr(plhs[0]);
    // sampling time for each query
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    double *duration = mxGetPr(plhs[1]);
    //---------------------
    // Start full search
    //---------------------
    mexPrintf("Starting Exhaustive Search for Queries......\n");mexEvalString("drawnow");
    size_t *maxIdx = (size_t*)malloc(NumMat * sizeof(size_t));
    maxIdx[0] = mxGetN(prhs[1]);
    maxIdx[1] = mxGetN(prhs[2]);
    SubIndex index(NumMat, maxIdx);
    progressbar(0);
    start = clock();
    for (size_t i = 0; i < NumQueries; ++i){
        clearprogressbar();
        progressbar(i/NumQueries);
        index.reset();
        std::list<double> listTop;
        for(size_t count = 0; count < knn && !index.isDone(); ++index){
            point3D p(i,index.getIdx()[0],index.getIdx()[1]);
            double temp = MatrixColMul(p,A,B,C);            
            listTop.push_back(temp);
            ++count;
        }
        listTop.sort();
        listTop.reverse();
        while(!index.isDone()){
            point3D p(i,index.getIdx()[0],index.getIdx()[1]);
            double temp = MatrixColMul(p,A,B,C); 
            if(temp > listTop.back()){
                doInsert(temp, listTop);
            }
            ++index;
        } 
        std::list<double>::iterator itr = listTop.begin();
        for(size_t p = 0; p < knn && p < listTop.size(); ++p){
            knnValue[i*knn + p] = *itr++;
        }
    }
    finish = clock();
    duration[0] = (double)(finish-start)/(NumQueries*CLOCKS_PER_SEC);
    progressbar(1);
    free(maxIdx);
}
