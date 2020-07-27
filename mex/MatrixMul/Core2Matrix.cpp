#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
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
  uint rankSize = mxGetN(prhs[0]);
  uint rankSizeExt = rankSize * rankSize;
  uint L_a = mxGetM(prhs[0]);
  uint L_b = mxGetM(prhs[1]);
  double *A = mxGetPr(prhs[0]);
  double *B = mxGetPr(prhs[1]);
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
  double *plhs_result = mxGetPr(plhs[0]);
  // time duration sampling
  plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  double *tsec = mxGetPr(plhs[1]);
  *tsec = duration;
  // indexes of values
  plhs[2] = mxCreateNumericMatrix(top_t, 2, mxUINT64_CLASS, mxREAL);
  uint64_T *plhs_pr = (uint64_T *)mxGetData(plhs[2]);
  mexPrintf("Starting Core^2 Sampling:");
  mexPrintf("Top:%d,Samples:1e%d,Budget:1e%d\n", top_t, (int)log10(NumSample),
            (int)log10(budget));
  mexEvalString("drawnow");
  // compute the extension for matrices
  start = clock();
  AT.transpose(A);
  BT.transpose(B);
  double *Aex = (double *)malloc(L_a * rankSizeExt * sizeof(double));
  double *Bex = (double *)malloc(L_b * rankSizeExt * sizeof(double));
  memset(Aex, 0, mxGetM(prhs[0]) * rankSizeExt * sizeof(double));
  memset(Bex, 0, mxGetM(prhs[1]) * rankSizeExt * sizeof(double));
  // compute the extension matrices
  for (uint m = 0; m < rankSize; ++m) {
    for (uint n = 0; n < rankSize; ++n) {
      size_t r = m * rankSize + n;
      // extension for matrix A
      double sum = 0;
      for (uint i = 0; i < L_a; ++i) {
        sum += abs(A[m * L_a + i] * A[n * L_a + i]);
        Aex[r * L_a + i] = sum;
      }
      // extension for matrix B
      sum = 0;
      for (uint j = 0; j < L_b; ++j) {
        sum += abs(B[m * L_b + j] * B[n * L_b + j]);
        Bex[r * L_b + j] = sum;
      }
    }
  }
  // extension matrices
  FactorMat MatAex(L_a, rankSizeExt, Aex, MATRIX_NONE_SUM);
  FactorMat MatBex(L_b, rankSizeExt, Bex, MATRIX_NONE_SUM);
  //-------------------------------------
  // Compute weight
  //-------------------------------------
  double SumofW = 0;
  double *weight = (double *)malloc(rankSizeExt * sizeof(double));
  memset(weight, 0, rankSizeExt * sizeof(double));
  for (uint r = 0; r < rankSizeExt; ++r) {
    weight[r] = MatAex(L_a - 1, r);
    weight[r] *= MatBex(L_b - 1, r);
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
  size_t SumCr = 0;
  size_t *freq_r = (size_t *)malloc(rankSizeExt * sizeof(size_t));
  memset(freq_r, 0, rankSizeExt * sizeof(size_t));
  for (uint r = 0; r < rankSizeExt; ++r) {
    double u = (double)rand() / (double)RAND_MAX;
    double c = (double)NumSample * weight[r] / SumofW;
    if (u < (c - floor(c)))
      freq_r[r] = (size_t)ceil(c);
    else
      freq_r[r] = (size_t)floor(c);
    SumCr += freq_r[r];
  }
  uint *IdxI = (uint *)malloc(SumCr * sizeof(uint));
  memset(IdxI, 0, SumCr * sizeof(uint));
  uint *IdxJ = (uint *)malloc(SumCr * sizeof(uint));
  memset(IdxJ, 0, SumCr * sizeof(uint));
  // sample indexes
  size_t offset = 0;
  for (uint r = 0; r < rankSizeExt; ++r) {
    binary_search(freq_r[r], (IdxI + offset), L_a, (MatAex.value + r * L_a));
    binary_search(freq_r[r], (IdxJ + offset), L_b, (MatBex.value + r * L_b));
    offset += freq_r[r];
  }
  // compute update value and saved in map<pair, value>
  // use map IrJc to save the sampled values
  // std::map<point2D, double> IrJc;
  Point2dValueMap IrJc;
  offset = 0;
  for (uint m = 0; m < rankSize; ++m) {
    for (uint n = 0; n < rankSize; ++n) {
      size_t r = m * rankSize + n;
      for (size_t s = 0; s < freq_r[r]; ++s, ++offset) {
        uint idxi = IdxI[offset];
        uint idxj = IdxJ[offset];
        int socre = sgn(AT(m, idxi)) * sgn(AT(n, idxi));
        socre *= sgn(BT(m, idxj)) * sgn(BT(n, idxj));
        IrJc[Point2d(idxi, idxj)] += socre;
      }
    }
  }
  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  *tsec += duration;
  mexPrintf("%f during the sampling phase.\n", duration);
  mexEvalString("drawnow");
  //-----------------------------------
  // sort the values have been sampled
  //-----------------------------------
  // for pre sort
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
  for (uint m = 0; m < sortVec.size() && m < top_t; ++m) {
    // value
    plhs_result[m] = sortVec[m].second;
    // i
    plhs_pr[m] = (sortVec[m].first.x + 1);
    // j
    plhs_pr[m + top_t] = (sortVec[m].first.y + 1);
  }
  mexPrintf("Done!\n");
  //---------------
  // free
  //---------------
  free(weight);
  free(IdxI);
  free(IdxJ);
  free(freq_r);
  free(Aex);
  free(Bex);
}
