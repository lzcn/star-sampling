#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <map>
#include <vector>

#include "include/matrix.h"
#include "include/utils.h"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  clock_t start, finish;
  double duration;
  srand(unsigned(time(NULL)));
  //--------------------
  // Initialization
  //--------------------
  double *pdf = mxGetPr(prhs[0]);
  uint size = (uint)mxGetM(prhs[0]);
  size_t NumSample = mxGetPr(prhs[1])[0];
  uint *dst = (uint *)malloc(NumSample * sizeof(uint));
  memset(dst, 0, NumSample * sizeof(uint));
  plhs[0] = mxCreateDoubleMatrix(NumSample, 1, mxREAL);
  double *result = mxGetPr(plhs[0]);
  plhs[1] = mxCreateDoubleMatrix(NumSample, 1, mxREAL);
  double *result1 = mxGetPr(plhs[1]);
  start = clock();
  binary_search(NumSample, dst, size, pdf);
  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  mexPrintf("%f during the binary_search.\n", duration);
  mexEvalString("drawnow");
  for (size_t i = 0; i < NumSample; i++) {
    result[i] = dst[i];
  }
  double sum = 0;
  for (size_t i = 0; i < size; i++) {
    sum += pdf[i];
  }
  start = clock();
  vose_alias(NumSample, dst, size, pdf, sum);
  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  mexPrintf("%f during the vose_alias.\n", duration);
  mexEvalString("drawnow");
  for (size_t i = 0; i < NumSample; i++) {
    result1[i] = dst[i];
  }
  free(dst);
}
