#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <map>
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
  // input
  start = clock();
  FactorMat MatA(mxGetM(prhs[0]), mxGetN(prhs[0]), mxGetPr(prhs[0]));
  FactorMat MatB(mxGetM(prhs[1]), mxGetN(prhs[1]), mxGetPr(prhs[1]));
  FactorMat MatC(mxGetM(prhs[2]), mxGetN(prhs[2]), mxGetPr(prhs[2]));
  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  const size_t budget = (size_t)mxGetPr(prhs[3])[0];
  const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
  const uint top_t = (uint)mxGetPr(prhs[5])[0];
  // output
  plhs[0] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
  double *plhs_result = mxGetPr(plhs[0]);
  plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  double *tsec = mxGetPr(plhs[1]);
  *tsec = duration;
  plhs[2] = mxCreateNumericMatrix(top_t, 3, mxUINT64_CLASS, mxREAL);
  uint64_T *plhs_pr = (uint64_T *)mxGetData(plhs[2]);
  mexPrintf("Starting Wedge Sampling:");
  mexPrintf("Top:%d,Samples:1e%d,Budget:1e%d\n", top_t, (int)log10(NumSample),
            (int)log10(budget));
  mexEvalString("drawnow");
  //-------------------------------------
  // Compute weight
  //-------------------------------------
  double SumofW = 0;
  // weight has the same size of A
  double *weight = (double *)malloc(MatA.n_row * MatA.n_col * sizeof(double));
  memset(weight, 0, MatA.n_row * MatA.n_col * sizeof(double));
  start = clock();
  for (uint r = 0; r < MatA.n_row; ++r) {
    for (uint i = 0; i < MatA.n_col; ++i) {
      double tempW = abs(MatA.GetElement(r, i));
      tempW *= MatB.col_abs_sum[r];
      tempW *= MatC.col_abs_sum[r];
      weight[r * MatA.n_col + i] = tempW;
      SumofW += tempW;
    }
  }
  for (uint r = 0; r < MatA.n_row; ++r) {
    for (uint i = 0; i < MatA.n_col; ++i) {
      weight[r * MatA.n_col + i] /= SumofW;
    }
  }
  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  *tsec += duration;
  //-------------------------
  // Do Sampling
  //-------------------------
  // sampled r, i, j, k
  uint *IdxI = (uint *)malloc(NumSample * sizeof(uint));
  memset(IdxI, 0, NumSample * sizeof(uint));
  uint *IdxJ = (uint *)malloc(NumSample * sizeof(uint));
  memset(IdxJ, 0, NumSample * sizeof(uint));
  uint *IdxK = (uint *)malloc(NumSample * sizeof(uint));
  memset(IdxK, 0, NumSample * sizeof(uint));
  uint *IdxR = (uint *)malloc(NumSample * sizeof(uint));
  memset(IdxR, 0, NumSample * sizeof(uint));
  // sampled r's frequency
  size_t *freq_r = (size_t *)malloc(MatA.n_row * sizeof(size_t));
  memset(freq_r, 0, MatA.n_row * sizeof(size_t));

  start = clock();
  // Do sample S pairs (r, i)
  sort_sample(NumSample, IdxI, IdxR, freq_r, MatA.n_row, MatA.n_col, weight, 1.0);
  // sample j,k;
  for (uint r = 0, offset = 0; r < MatA.n_row; ++r) {
    vose_alias(freq_r[r], (IdxJ + offset), MatB.n_row,
               (MatB.value + r * MatB.n_row), MatB.col_abs_sum[r]);
    vose_alias(freq_r[r], (IdxK + offset), MatC.n_row,
               (MatC.value + r * MatC.n_row), MatC.col_abs_sum[r]);
    offset += freq_r[r];
  }
  // compute update value and saved in map<pair, value>
  // use map IrJc to save the sampled values
  std::map<Point3d, double> IrJc;
  for (size_t s = 0; s < NumSample; ++s) {
    uint i = IdxI[s];
    uint j = IdxJ[s];
    uint k = IdxK[s];
    uint r = IdxR[s];
    double valueSampled = sgn_foo(MatA.GetElement(r, i)) *
                          sgn_foo(MatB.GetElement(j, r)) *
                          sgn_foo(MatC.GetElement(k, r));
    // Update the element in coordinate
    IrJc[Point3d(i, j, k)] += valueSampled;
  }
  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  *tsec += duration;
  //-----------------------------------
  // sort the values have been sampled
  //-----------------------------------
  std::vector<Point3dValuePair> tempSortedVec;
  std::vector<Point3dValuePair> sortVec;
  std::map<Point3d, double>::iterator mapItr;
  for (mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr) {
    tempSortedVec.push_back(std::make_pair(mapItr->first, mapItr->second));
  }
  start = clock();
  sort(tempSortedVec.begin(), tempSortedVec.end(), compgt<Point3dValuePair>);
  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  *tsec += duration;
  start = clock();
  for (size_t m = 0; m < tempSortedVec.size() && m < budget; ++m) {
    double true_value = vectors_mul(tempSortedVec[m].first, MatA, MatB, MatC);
    sortVec.push_back(std::make_pair(tempSortedVec[m].first, true_value));
  }
  sort(sortVec.begin(), sortVec.end(), compgt<Point3dValuePair>);

  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  *tsec += duration;
  //--------------------------------
  // Converting to Matlab
  //--------------------------------
  for (uint m = 0; m < sortVec.size() && m < top_t; ++m) {
    // value
    plhs_result[m] = sortVec[m].second;
    // i
    plhs_pr[m] = (sortVec[m].first.x + 1);
    // j
    plhs_pr[m + top_t] = (sortVec[m].first.y + 1);
    // k
    plhs_pr[m + top_t + top_t] = (sortVec[m].first.z + 1);
  }
  mexPrintf("Done\n");
  //---------------
  // free
  //---------------
  free(weight);
  free(IdxI);
  free(IdxJ);
  free(IdxK);
  free(IdxR);
  free(freq_r);
}
