/*
    exhaustive search for three order tensor given factor matrices
    factor matrices has the same row dimension, which is the 
    dimension of feature vector.
    [value, time] = exact_search(A1,A1,A3,top_t)
    it will return the top_t value in the tensor,
    and the time during computing
    Author : Zhi Lu
*/

#include <list>
#include <ctime>
#include <map>

#include "mex.h"
#include "matrix.h"

typedef std::pair<point3D,double> indValue;

int cmp(const indValue &x,const indValue&y){
    return (x.second > y.second);
}

/*
    all matrices must has the same row dimension
    [value, time, indexes] = exactSearchThreeOrderTrensor(A,B,C,top_t)
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
    std::list<point3D> listIdx;
    std::vector<indValue> tempVec;

    const int top_t = mxGetPr(prhs[3])[0];
    double temp = 0.0;
    bool initTop = true;
    int count  = 0;
    int si,sj,sk;
    for (int i = 0; i < A.col && initTop; ++i){
        for (int j = 0; j < B.col && initTop; ++j){
            for (int k = 0; k < C.col && initTop; ++k){
                temp = MatrixColMul(A,B,C,i,j,k);
                if(count < top_t){
                    tempVec.push_back(std::make_pair(point3D(i,j,k),temp));
                }else{
                    initTop = false;
                    si = i;
                    sj = j;
                    sk = k;
                }
                ++count;
            }
        }
    }
    sort(tempVec.begin(),tempVec.end(),cmp);
    for(auto itr = tempVec.begin(); itr != tempVec.end(); ++itr){
        listTop.push_back(itr->second);
        listIdx.push_back(itr->first);
    }
    start = clock();
    for(int i = si; i < A.col; ++i){
        for(int j = sj; j < B.col; ++j){
            for(int k = sk; k < C.col; ++k){
                temp = MatrixColMul(A,B,C,i,j,k);
                if(temp > listTop.back()){
                    doInsert(temp, listTop, point3D(i,j,k), listIdx);
                }
            }
        }
    }
    finish = clock();
    duration = (double)(finish - start)/CLOCKS_PER_SEC;
    //---------------------------------
    // convert result to Matlab format
    //---------------------------------
    size_t length = listTop.size();
    plhs[0] = mxCreateDoubleMatrix(length,1,mxREAL);
    double *topValue = mxGetPr(plhs[0]);
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    mxGetPr(plhs[1])[0] = duration;
    plhs[2] = mxCreateNumericMatrix(length, 3, mxUINT64_CLASS, mxREAL);
    uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[2]);  
    auto itr = listTop.begin();
    auto itr2 = listIdx.begin();
    for(int i = 0; i < length; ++i){
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
}