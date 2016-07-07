#include <list>
#include <vector>
#include <algorithm>

#include "mex.h"
#include "../../include/matrix.h"

/*coordinate to save the result*/

void disp_coord(const size_t* cur_ind,int nrhs){
    for(int i = 0; i < nrhs; ++i){
        printf("%d, ", cur_ind[i]);
    }
    printf("\n");

}

double getValue(const size_t *curIdx, \
                std::vector<double*> &vecMat, \
                size_t rank, int num ){
    // temp value
    double *temp = (double*)malloc(rank*sizeof(double));
    memset(temp, 1, rank*sizeof(double));
    // the feature vector 
    double *feature;
    for (int i = 0; i < num; ++i){
        // element-wise multiplication
        // the address of curIdx[i]-th feature of i-th matrix
        feature = (vecMat[i] + curIdx[i]*rank);
        for(size_t j = 0; j < rank; ++j){
            temp[j] *= *(feature+j);
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
    usage: [index, result] = exactSearch(A,B[,C,...], top_t)
    find the top_t elements in tensor
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   
    //---------------------
    // initialization
    //---------------------

    const int rank = mxGetM(prhs[0]);
    const int top_t = (int)mxGetPr(prhs[nrhs-1])[0];
    std::vector<double*> vecMat;
    size_t *max = (size_t *)malloc((nrhs - 1)*sizeof(size_t));
    memset(max, 0, (nrhs - 1)*sizeof(size_t));
    // matrices
    for(int i = 0; i < nrhs - 1; ++i){
        vecMat.push_back(mxGetPr(prhs[i]));
        max[i] = mxGetN(prhs[i]);
    }
    //------------------------
    // Do exhaustive computing
    //------------------------
    std::list<double> listTop;
    // subIndex for loop
    SubIndex subIndex(nrhs - 1, max);
    int count  = 0;
    double tempValue = 0.0;
    // compute top_t values as the initial list
    while(!subIndex.isDone() && count < top_t){
        tempValue = getValue(subIndex.getIdx(), \
                             vecMat, rank, nrhs-1);
        listTop.push_back(tempValue);
    }
    // sort the list in descending order
    listTop.sort();
    listTop.reverse();
    // do exhaustive search
    subIndex.reset();
    while(!subIndex.isDone()){
        tempValue = getValue(subIndex.getIdx(),vecMat,rank,nrhs-1);
        if(tempValue > listTop.back()){
            doInsert(tempValue, listTop);
        }
        ++subIndex;
    }
    //-----------------------------
    // convert the result to Matlab
    //-----------------------------
    plhs[0] = mxCreateDoubleMatrix(listTop.size(), 1, mxREAL);
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