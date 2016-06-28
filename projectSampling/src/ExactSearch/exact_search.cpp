
#include <list>
#include <vector>
#include <algorithm>

#include "mex.h"
#include "../../matrix.h"

/*coordinate to save the result*/

void disp_coord(const size_t* cur_ind,int nrhs){
    for(int i = 0; i < nrhs; ++i){
        printf("%d, ", cur_ind[i]);
    }
    printf("\n");

}

double getValue( const size_t *curIdx, \
                    std::vector<double*> &vecMat, \
                    size_t rank,int num ){
    double ans = 0;
    double *temp = new double[rank];
    /*initialize to 1 */
    for (size_t i = 0; i < rank; ++i){
        temp[i] = 1;
    }
    double *p;
    for (int i = 0; i < num; ++i){
        /*element-wise multiplication*/
        p = (vecMat[i] + curIdx[i]*rank);
        for(size_t j = 0; j < rank; ++j){
            temp[j] *= *(p+j);
        }
    }
    for (size_t i = 0; i < rank; ++i){
        ans += temp[i];
    }
    delete []temp;
    return ans;
}


int doInsert(double value,double*toInsert,size_t top_t){
    double front,next;
    for(size_t i = 0; i < top_t; ++i){
        if(value > toInsert[i]){
            // find and insert
            front = toInsert[i];
            toInsert[i] = value;
            // shift the left element
            for(int j = (i + 1); j < top_t; ++j){
                next = toInsert[j];
                toInsert[j] = front;
                front = next;
            }
            return i;
        }
    }
    return top_t;
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
    const size_t rank = mxGetM(prhs[0]);
    const int top_t = (int)mxGetPr(prhs[nrhs-1])[0];
    std::vector<double*> vecMat;
    size_t *max = (size_t *);malloc((nrhs - 1)*sizeof(size_t));
    memset(max, 0, (nrhs - 1)*sizeof(size_t));
    // matrices
    for(int i = 0; i < nrhs - 1; ++i){
        vecMat.push(mxGetPr(prhs[i]));
        max[i] = mxGetN(prhs[i]);
    }
    // subIndex for loop
    SubIndex subIndex(nrhs - 1, max);
    //------------------------
    // Do exhaustive computing
    //------------------------
    std::list<double> listTop;
    int count  = 0;
    while(!subIndex.isdone()){
        temp_value = getValue(subIndex.get_ind(),vecMat,rank,nrhs);
        if(temp_value > max_value[top_t]){
            doInsert(temp_value,max_value,top_t);
        }
        ++count;
    }
    double temp_value = 0;
    double init_val = getValue(subIndex.get_ind(),vecMat,rank,nrhs);
    double *max_value = new double[top_t];
    double max_v = 0.0;
    while(!subIndex.isdone()){
        temp_value = getValue(subIndex.get_ind(),vecMat,rank,nrhs);
        if(temp_value > max_value[top_t]){
            doInsert(temp_value,max_value,top_t);
        }
        ++subIndex;
    }
    delete []p;
    plhs[0] = mxCreateDoubleMatrix(top_t,1,mxREAL);
    double *plhs_max;
    plhs_max = mxGetPr(plhs[0]);
    for(size_t i = 0; i < top_t; ++i){
        plhs_max[i] = max_value[i];
    }
    delete []max_value;

}