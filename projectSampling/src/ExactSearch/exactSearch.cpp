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
#include "../../include/matrix.h"

/*
    Compute the value in coordinate
*/
double getValue(const size_t *curIdx, \
                const std::vector<double*> &vecMat, \
                size_t rank, size_t numMat ){
    // temp value
    double *temp = (double*)malloc(rank*sizeof(double));
    for (int i = 0; i < rank; ++i){
        temp[i] = 1.0;
    }
    // the feature vector 
    double *feature;
    for (int i = 0; i < numMat; ++i){
        // element-wise multiplication
        // the address of curIdx[i]-th feature of i-th matrix
        feature = &vecMat[i][curIdx[i]*rank];
        for(size_t j = 0; j < rank; ++j){
            temp[j] *= vecMat[i][curIdx[i]*rank+j];
        }
    }
    double result = 0;
    for (size_t i = 0; i < rank; ++i){
        result += temp[i];
    }
    delete []temp;
    return result;
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
    const int top_t = (int)mxGetPr(prhs[nrhs-1])[0];
    size_t *max = (size_t *)malloc(numMat*sizeof(size_t));
    memset(max, 0, numMat*sizeof(size_t));
    // matrices
    std::vector<double*> vecMat;
    for(int i = 0; i < numMat; ++i){
        vecMat.push_back(mxGetPr(prhs[i]));
        max[i] = mxGetN(prhs[i]);
    }
    //------------------------
    // Do exhaustive computing
    //------------------------
    std::list<double> listTop;
    // subIndex for loop
    SubIndex index(numMat, max);
    int count  = 0;
    double tempValue = 0.0;
    // compute top_t values as the initial list
    while(!index.isDone() && count < top_t){
        tempValue = getValue(index.getIdx(),vecMat,rank,numMat);
        
        listTop.push_back(tempValue);
        ++index;
        ++count;
    }
    // sort the list in descending order
    listTop.sort();
    listTop.reverse();
    // do exhaustive search
    start = clock();
    while(!index.isDone()){
        tempValue = getValue(index.getIdx(),vecMat,rank,nrhs-1);
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
    for(int i = 0; i < listTop.size(); ++i){
        topValue[i] = (*itr);
        ++itr;
    }
    //-------------------
    // free
    //--------------------
    free(max);

}