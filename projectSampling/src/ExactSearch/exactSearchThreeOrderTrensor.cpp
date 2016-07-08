#include <list>
#include <ctime>

#include "mex.h"
#include "../../include/matrix.h"


/*
    all matrices must has the same row dimension
    [value, time] = exactSearchThreeOrderTrensor(A,B,C,top_t)
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   

    clock_t start, finish;
    double duration;
    const int rank_size = mxGetM(prhs[0]);

    Matrix A(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
    Matrix B(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
    Matrix C(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));

    std::list<double> listTop;

    const int top_t = mxGetPr(prhs[3])[0];
    double temp = 0.0;
    bool initTop = true;
    int count  = 0;
    for (int i = 0; i < A.col && initTop; ++i){
        for (int j = 0; j < B.col && initTop; ++j){
            for (int k = 0; k < C.col && initTop; ++k){
                temp = MatrixColMul(A,B,C,i,j,k);
                if(count < top_t){
                    listTop.push_back(temp);
                }else{
                    initTop = false;
                }
                ++count;
            }
        }
    }
    listTop.sort(); listTop.reverse();
    start = clock();
    for(int i = 0; i < A.col; ++i){
        for(int j = 0; j < B.col; ++j){
            for(int k = 0; k < C.col; ++k){
                temp = MatrixColMul(A,B,C,i,j,k);
                if(temp > listTop.back()){
                    doInsert(temp, listTop);
                }
            }
        }
    }
    finish = clock();
    duration = (double)(finish - start)/CLOCKS_PER_SEC;
    plhs[0] = mxCreateDoubleMatrix(listTop.size(),1,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    mxGetPr(plhs[1])[0] = duration;
    double *topValue = mxGetPr(plhs[0]);
    std::list<double>::iterator itr = listTop.begin();
    for(int i = 0; i < listTop.size(); ++i){
        topValue[i] = (*itr);
        ++itr;
    }
}