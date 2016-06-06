#include <list>
#include "mex.h"
#include "../../include/matrix.h"


/*
    the matrix must has the same row dimension
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   

    const int rank_size = mxGetM(prhs[0]);

    Matrix A(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
    Matrix B(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
    Matrix C(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));

    std::list<float> listTop;

    const int top_t = 1000;
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
    plhs[0] = mxCreateDoubleMatrix(listTop.size(),1,mxREAL);
    double *topValue = mxGetPr(plhs[0]);
    for(int i = 0; i < listTop.size(); ++i){
        topValue[i] = listTop[i];
    }
}