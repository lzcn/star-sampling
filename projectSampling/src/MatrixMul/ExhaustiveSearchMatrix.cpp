/*
    ExhaustiveSearch(A,B,top_t,TYPE)
    A,B same row dimension
    TYPE:{"Euclidean","Cosine"}
*/
#include <list>
#include <ctime>
#include <cstring>
#include <iostream>
#include "mex.h"
#include "matrix.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   
    clock_t start, finish;
    double *duration;
    double (*metric)(const point2D&, const Matrix&, const Matrix&);
    //--------------------------
    // initialization
    //--------------------------
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    duration = mxGetPr(plhs[1]);
    size_t NumMats = 2;
    Matrix A(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
    Matrix B(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
    const int top_t = mxGetPr(prhs[2])[0];
    char *type;
    type = mxArrayToString(prhs[3]);
    if(!strcmp(type,"Euclidean")){
        metric = EuclideanMetric;
        mexPrintf("Using Ranking Function:Euclidean Metric...");
    }else if(!strcmp(type,"Cosine")){
        metric = CosineMetric;
        mexPrintf("Using Ranking Function:Cosine Similarity...");
    }else{
        mexPrintf("No Match Ranking Functions...\n");
        return;
    }
    // loop index
    size_t *max = (size_t*)malloc(NumMats*sizeof(size_t));
    for (int i = 0; i < NumMats; ++i){
        max[i] = mxGetN(prhs[i]);
    }
    SubIndex index(NumMats, max);
    start = clock();
    std::list<double> listTop;
    std::vector<double> tempVec;
    for(size_t count = 0; count < top_t && !index.isDone(); ++index){
        double temp = metric(point2D(index.getIdx()[0],index.getIdx()[1]), A, B);
        tempVec.push_back(temp);
        ++count;
    }
    sort(tempVec.begin(),tempVec.end());
    for(auto itr = tempVec.begin(); itr != tempVec.end(); ++itr){
        listTop.push_back(*itr);
    }
    if(!strcmp(type,"Cosine")){
        listTop.reverse();
        while(!index.isDone()){
            double temp = metric(point2D(index.getIdx()[0],index.getIdx()[1]), A, B);
            if(temp > listTop.back()){
                doInsert(temp, listTop);
            }
            ++index;
        }
    }else{
        while(!index.isDone()){
            double temp = metric(point2D(index.getIdx()[0],index.getIdx()[1]), A, B);
            if(temp < listTop.back()){
                doInsertReverse(temp, listTop);
            }
            ++index;
        }
    }
    finish = clock();
    duration[0] = (double)(finish - start)/CLOCKS_PER_SEC;
    //---------------------------------
    // convert result to Matlab format
    //---------------------------------
    size_t length = listTop.size();
    plhs[0] = mxCreateDoubleMatrix(length,1,mxREAL);
    double *topValue = mxGetPr(plhs[0]);
    auto itr = listTop.begin();
    for(size_t i = 0; i < length; ++i){
        topValue[i] = *itr++;
    }
    mexPrintf("Done!\n");
    free(max);
    mxFree(type);
}
