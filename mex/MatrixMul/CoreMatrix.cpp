#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <vector>

#include "matrix.h"
#include "mex.h"
#include "utils.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  clock_t start, finish;
  double duration;
  srand(unsigned(time(NULL)));
  //--------------------
  // Initialization
  //--------------------
  uint rankSize = mxGetN(prhs[0]);
  uint L_a = (uint)mxGetM(prhs[0]);
  uint L_b = (uint)mxGetM(prhs[1]);
  FactorMat MatA(L_a, rankSize);
  FactorMat MatB(L_b, rankSize);
  FactorMat AT(rankSize, L_a);
  FactorMat BT(rankSize, L_b);
  // the budget
  const size_t budget = (size_t)mxGetPr(prhs[2])[0];
  // number of samples
  const size_t NumSample = (size_t)mxGetPr(prhs[3])[0];
  // find the top-t largest value
  const uint top_t = (uint)mxGetPr(prhs[4])[0];
  // result of sampling
  plhs[0] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
  double *values = mxGetPr(plhs[0]);
  // time duration sampling
  plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  double *tsec = mxGetPr(plhs[1]);
  *tsec = duration;
  // indexes of values
  plhs[2] = mxCreateNumericMatrix(top_t, 2, mxUINT64_CLASS, mxREAL);
  uint64_T *indexes = (uint64_T *)mxGetData(plhs[2]);
  mexPrintf("Starting Core^1 Sampling:");
  mexPrintf("Top:%d,Samples:1e%d,Budget:1e%d\n", top_t, (int)log10(NumSample),
            (int)log10(budget));
  mexEvalString("drawnow");
  //-------------------------------------
  // Initialization
  //-------------------------------------
  start = clock();
  MatA.accumulation(mxGetPr(prhs[0]));
  MatB.accumulation(mxGetPr(prhs[1]));
  AT.transpose(mxGetPr(prhs[0]));
  BT.transpose(mxGetPr(prhs[1]));
  double SumofW = 0;
  double *weight = (double *)malloc(rankSize * sizeof(double));
  memset(weight, 0, rankSize * sizeof(double));
  for (uint r = 0; r < rankSize; ++r) {
    weight[r] = MatA(L_a - 1, r);
    weight[r] *= MatA(L_b - 1, r);
    SumofW += weight[r];
  }
  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  *tsec = duration;
  mexPrintf("%f during the initialization phase.\n", duration);
  mexEvalString("drawnow");
  //-------------------------
  // Do Sampling
  //-------------------------
  start = clock();
  size_t TotalS = 0;
  size_t *freq_r = (size_t *)malloc(rankSize * sizeof(size_t));
  memset(freq_r, 0, rankSize * sizeof(size_t));
  for (uint r = 0; r < rankSize; ++r) {
    double u = (double)rand() / (double)RAND_MAX;
    double c = (double)NumSample * weight[r] / SumofW;
    if (u < (c - floor(c)))
      freq_r[r] = (size_t)ceil(c);
    else
      freq_r[r] = (size_t)floor(c);
    TotalS += freq_r[r];
  }
  uint *IdxI = (uint *)malloc(TotalS * sizeof(uint));
  memset(IdxI, 0, TotalS * sizeof(uint));
  uint *IdxJ = (uint *)malloc(TotalS * sizeof(uint));
  memset(IdxJ, 0, TotalS * sizeof(uint));
  // sample indexes
  size_t offset = 0;
  for (uint r = 0; r < rankSize; ++r) {
    binary_search(freq_r[r], (IdxI + offset), L_a, (MatA.value + r * L_a));
    binary_search(freq_r[r], (IdxJ + offset), L_b, (MatB.value + r * L_b));
    offset += freq_r[r];
  }
  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  *tsec += duration;
  mexPrintf("%f during the sampling phase.\n", duration);
  mexEvalString("drawnow");
  //------------
  // Filtering
  //------------
  start = clock();
  Point2dValueMap IrJc;
  offset = 0;
  for (uint r = 0; r < rankSize; ++r) {
    for (size_t s = 0; s < freq_r[r]; ++s) {
      uint idxi = IdxI[offset];
      uint idxj = IdxJ[offset];
      IrJc[Point2d(idxi, idxj)] += sgn(AT(r, idxi)) * sgn(BT(r, idxj));
      ++offset;
    }
  }
  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  *tsec += duration;
  mexPrintf("%f during the scoring phase.\n", duration);
  mexEvalString("drawnow");
  //-----------------------------------
  // sort the values have been sampled
  //-----------------------------------
  // for pre-sort
  start = clock();
  std::vector<Point2dValuePair> tempSortedVec;
  std::vector<Point2dValuePair> sortVec;
  // push the value into a vector for sorting
  for (auto mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr) {
    tempSortedVec.push_back(std::make_pair(mapItr->first, mapItr->second));
  }
  sort(tempSortedVec.begin(), tempSortedVec.end(), compgt<Point2dValuePair>);
  for (size_t m = 0; m < tempSortedVec.size() && m < budget; ++m) {
    double true_value = MatrixColMul(tempSortedVec[m].first, AT, BT);
    sortVec.push_back(std::make_pair(tempSortedVec[m].first, true_value));
  }
  // sort the vector according to the actual value
  sort(sortVec.begin(), sortVec.end(), compgt<Point2dValuePair>);
  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  *tsec += duration;
  mexPrintf("%f during the sorting phase.\n", duration);
  mexEvalString("drawnow");
  //--------------------------------
  // Converting to Matlab
  //--------------------------------
  // value
  for (size_t m = 0; m < sortVec.size() && m < top_t; ++m) {
    // value
    values[m] = sortVec[m].second;
    // i
    indexes[m] = (sortVec[m].first.x + 1);
    // j
    indexes[m + top_t] = (sortVec[m].first.y + 1);
  }
  mexPrintf("Done!\n");
  //---------------
  // free
  //---------------
  free(IdxI);
  free(IdxJ);
  free(freq_r);
  free(weight);
}
