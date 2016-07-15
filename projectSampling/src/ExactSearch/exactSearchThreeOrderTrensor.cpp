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
    double *duration;
    const int rank_size = mxGetM(prhs[0]);
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    duration = mxGetPr(plhs[1]);
    Matrix A(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
    Matrix B(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
    Matrix C(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));

    std::list<double> listTop;
    std::list<point3D> listIdx;
    std::vector<indValue> tempVec;

    const int top_t = mxGetPr(prhs[3])[0];
    double temp = 0.0;
    size_t *max = (size_t*)malloc(3*sizeof(size_t));
    for (int i = 0; i < 3; ++i){
        max[i] = mxGetN(prhs[i]);
    }
    start = clock();
    SubIndex index(3,max);
    for(size_t count = 0; count < top_t && !index.isDone(); ++index){
        temp = MatrixColMul(A,B,C,index.getIdx()[0],index.getIdx()[1],index.getIdx()[2]);
        tempVec.push_back(std::make_pair(point3D(index.getIdx()[0],index.getIdx()[1],index.getIdx()[2]),temp));
        ++count;
    }
    sort(tempVec.begin(),tempVec.end(),cmp);
    for(auto itr = tempVec.begin(); itr != tempVec.end(); ++itr){
        listTop.push_back(itr->second);
        listIdx.push_back(itr->first);
    }
    while(!index.isDone()){
        temp = MatrixColMul(A,B,C,index.getIdx()[0],index.getIdx()[1],index.getIdx()[2]);
        if(temp > listTop.back()){
            doInsert(temp, listTop, point3D(index.getIdx()[0],index.getIdx()[1],index.getIdx()[2]), listIdx);
        }
        ++index;
    }
    finish = clock();
    duration[0] = (double)(finish - start)/CLOCKS_PER_SEC;
    //---------------------------------
    // convert result to Matlab format
    //---------------------------------
    size_t length = listTop.size();
    plhs[0] = mxCreateDoubleMatrix(length,1,mxREAL);
    double *topValue = mxGetPr(plhs[0]);
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
    free(max);
}