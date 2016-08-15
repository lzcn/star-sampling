/*
    exhaustive search for tensor given factor matrices
    factor matrices has the same row size, which is the 
    dimension of feature vector.
    [value, time] = exact_search(A1,A1,...,AN,top_t)
    it will return the top_t value in the tensor,
    and the time during computing
    Author : Zhi Lu
*/

#include <list>
#include <vector>
#include <algorithm>
#include <ctime>

#include "mex.h"
#include "matrix.h"

double ColMul(const size_t *curIdx, double **p, size_t rank, size_t numMat){
    double ans = 0.0;
    for(size_t r = 0; r < rank; ++r){
        double temp = 1.0;
        for(size_t i = 0; i < numMat; ++i){
            temp *= p[i][curIdx[i] * rank + r];
        }
        ans += temp;
    }
    return ans;
}
/*
    matrices has the same row size
    find the top_t elements in tensor
    [value, time] = exact_search(A1,A1,...,AN,top_t)
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   
    clock_t start, finish;
    //---------------------
    // initialization
    //---------------------
    double duration;
    const size_t rank = mxGetM(prhs[0]);
    const size_t numMat = nrhs - 1;
    const size_t top_t = (size_t)mxGetPr(prhs[nrhs-1])[0];
    size_t *max = (size_t *)malloc(numMat*sizeof(size_t));
    memset(max, 0, numMat*sizeof(size_t));
    double **Mats = (double**)malloc(numMat*sizeof(double*));
    for(size_t i = 0; i < numMat; ++i){
        Mats[i] = mxGetPr(prhs[i]);
        max[i] = mxGetN(prhs[i]);
    }
    //------------------------
    // Do exhaustive computing
    //------------------------
    std::list<double> listTop;
    // subIndex for loop
    start = clock();
    SubIndex index(numMat, max);
    // compute top_t values as the initial list
    for(size_t count = 0; count < top_t && !index.isDone(); ++index,++count){
        double tempValue = ColMul(index.getIdx(),Mats,rank,numMat);
        listTop.push_back(tempValue);
    }
    // sort the list in descending order
    listTop.sort();
    listTop.reverse();
    // do exhaustive search
    while(!index.isDone()){
        double tempValue = ColMul(index.getIdx(),Mats,rank,numMat);
        if(tempValue > listTop.back()){
            doInsert(tempValue, listTop);
        }
        ++index;
    }
    finish = clock();
    duration = (double)(finish - start)/CLOCKS_PER_SEC;
    //-----------------------------
    // convert the result to Matlab
    //-----------------------------
    plhs[0] = mxCreateDoubleMatrix(listTop.size(), 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    mxGetPr(plhs[1])[0] = duration;
    double *topValue = mxGetPr(plhs[0]);
    std::list<double>::iterator itr = listTop.begin();
    for(size_t i = 0; i < listTop.size(); ++i){
        topValue[i] = *itr++;
    }
    //-------------------
    // free
    //--------------------
    free(max);
    free(Mats);

}
