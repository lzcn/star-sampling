#include <list>
#include <ctime>
#include <cstring>
#include <iostream>
#include "mex.h"
#include "matrix.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   
    
    //check for proper number of arguments
    if(nrhs != 4)
        mexErrMsgIdAndTxt( "MATLAB:revord:invalidNumInputs", "Four input required.");
    else if(nlhs > 4) 
        mexErrMsgIdAndTxt( "MATLAB:revord:maxlhs", "Too many output arguments.");
    if ( mxIsChar(prhs[3]) != 1)
        mexErrMsgIdAndTxt( "MATLAB:revord:inputNotString", "Fourth Input must be a string.");
    srand(unsigned(time(NULL)));
    double *duration;
    clock_t start, finish;
    double (*metric)(const point2D&, const Matrix&, const Matrix&);
    void (*insert)(double p, std::list<double> &listTop);
    //--------------------------
    // initialization
    //--------------------------
    const size_t rankSize = mxGetM(prhs[0]);
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
        insert = doInsertReverse;
        mexPrintf("Using Ranking Function:Euclidean Metric...\n");
    }else if(!strcmp(type,"Cosine")){
        metric = CosineMetric;
        insert = doInsert;
        mexPrintf("Using Ranking Function:Cosine Similarity...\n");
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
    }
    while(!index.isDone()){
        double temp = metric(point2D(index.getIdx()[0],index.getIdx()[1]), A, B);
        if(temp > listTop.back()){
            insert(temp, listTop);
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
    auto itr = listTop.begin();
    for(int i = 0; i < length; ++i){
        topValue[i] = *itr++;
    }
    free(max);
    mxFree(type);
}
