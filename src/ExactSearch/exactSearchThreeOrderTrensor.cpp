#include <list>
#include <ctime>

#include "mex.h"
#include "matrix.h"
#include "utilmex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   
    clock_t start, finish;
    // output for time consuming
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    double *duration = mxGetPr(plhs[1]);
    const uint L_a  = (uint)mxGetN(prhs[0]);
    const uint L_b  = (uint)mxGetN(prhs[1]);
    const uint L_c  = (uint)mxGetN(prhs[2]);
    const uint rankSize = (uint)mxGetM(prhs[0]);
    const uint top_t = (uint)mxGetPr(prhs[3])[0];
    const uint numMat = 3;
    // matrices
    Matrix A(rankSize, L_a, mxGetPr(prhs[0]), MATRIX_NONE_SUM);
    Matrix B(rankSize, L_b, mxGetPr(prhs[1]), MATRIX_NONE_SUM);
    Matrix C(rankSize, L_c, mxGetPr(prhs[2]), MATRIX_NONE_SUM);
    // varibales for progress bar
    double total = mxGetN(prhs[0])*mxGetN(prhs[1])*mxGetN(prhs[2]);
    double progress = 0;
    uint flag = 0;
    // list for top values and coordinates
    std::list<double> listTop;
    std::list<point3D> listIdx;
    std::vector<pidx3d> tempVec;
    // counter for coordinates
    uint *max = (uint*)malloc(numMat*sizeof(uint));
    memset(max, 0, numMat*sizeof(uint));
    for (int i = 0; i < numMat; ++i){
        max[i] = (uint)mxGetN(prhs[i]);
    }
    SubIndex index(numMat, max);
    mexPrintf("Start Exhaustive Search...\n");
    mexEvalString("drawnow");
    progressbar(0);
    start = clock();
    for(uint count = 0; count < top_t && !index.isDone(); ++count){
        double temp = MatrixColMul(A, B, C,
                        index.getIdx()[0], index.getIdx()[1], index.getIdx()[2]);
        tempVec.push_back(std::make_pair(
                                point3D(index.getIdx()[0],
                                        index.getIdx()[1],
                                        index.getIdx()[2]),temp));
        ++index;
        progress += 1;
    }
    sort(tempVec.begin(), tempVec.end(), compgt<pidx3d>);
    for(auto itr = tempVec.begin(); itr != tempVec.end(); ++itr){
        listTop.push_back(itr->second);
        listIdx.push_back(itr->first);
    }
    while(!index.isDone()){
        double temp = MatrixColMul(A, B, C,
                        index.getIdx()[0],
                        index.getIdx()[1],
                        index.getIdx()[2]);
        if(temp > listTop.back()){
            doInsert(temp, listTop, 
                point3D(index.getIdx()[0],index.getIdx()[1],index.getIdx()[2]), 
                listIdx);
        }
        ++index;
        progress += 1;
        flag += 1;
        if(flag > 1e8){
            clearprogressbar();
            progressbar(progress/total);
            flag = 0;
        }
    }
    finish = clock();
    duration[0] = (double)(finish - start)/CLOCKS_PER_SEC;
    //---------------------------------
    // convert result to Matlab format
    //---------------------------------
    uint length = listTop.size();
    plhs[0] = mxCreateDoubleMatrix(length,1,mxREAL);
    double *topValue = mxGetPr(plhs[0]);
    plhs[2] = mxCreateNumericMatrix(length, 3, mxUINT64_CLASS, mxREAL);
    uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[2]);  
    auto itr = listTop.begin();
    auto itr2 = listIdx.begin();
    for(uint i = 0; i < length; ++i){
        // value
        topValue[i] = (*itr);
        // indexes
        plhs_pr[i] = ((*itr2).x + 1);
        //j
        plhs_pr[i + length] = ((*itr2).y + 1);
        //k
        plhs_pr[i + length + length] = ((*itr2).z + 1);
        ++itr;
        ++itr2;
    }
    clearprogressbar();
    progressbar(1);
    mexPrintf("\n");
    free(max);
}
