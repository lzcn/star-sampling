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
  clock_t start;
  srand(unsigned(time(NULL)));
  //--------------------
  // Initialization
  //--------------------
  uint rankSize = (uint)mxGetN(prhs[0]);
  uint rankSizeExt = rankSize * rankSize;
  // original matrices
  double *A = mxGetPr(prhs[0]);
  double *B = mxGetPr(prhs[1]);
  double *C = mxGetPr(prhs[2]);
  uint L_a = (uint)mxGetM(prhs[0]);
  uint L_b = (uint)mxGetM(prhs[1]);
  uint L_c = (uint)mxGetM(prhs[2]);
  // to save the transpose
  FactorMat AT(rankSize, L_a);
  FactorMat BT(rankSize, L_b);
  FactorMat CT(rankSize, L_c);
  // the budget
  const size_t budget = (size_t)mxGetPr(prhs[3])[0];
  // number of samples
  const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
  // find the top-t largest value
  const uint top_t = (uint)mxGetPr(prhs[5])[0];
  // result of sampling
  plhs[0] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
  double *values = mxGetPr(plhs[0]);
  // time duration sampling
  plhs[1] = mxCreateDoubleMatrix(1, 4, mxREAL);
  double *tsec = mxGetPr(plhs[1]);
  // indexes of values
  plhs[2] = mxCreateNumericMatrix(top_t, 3, mxUINT64_CLASS, mxREAL);
  uint64_T *indexes = (uint64_T *)mxGetData(plhs[2]);
  mexPrintf("Starting Core^2 Sampling:");
  mexPrintf("Top:%d,Samples:1e%d,Budget:1e%d\n", top_t, (int)log10(NumSample),
            (int)log10(budget));
  mexEvalString("drawnow");
  // compute the extension for matrices
  Timer Time_;
  start = clock();
  // do transpose
  AT.transpose(A);
  BT.transpose(B);
  CT.transpose(C);
  double *A_ = (double *)malloc(L_a * rankSizeExt * sizeof(double));
  double *B_ = (double *)malloc(L_b * rankSizeExt * sizeof(double));
  double *C_ = (double *)malloc(L_c * rankSizeExt * sizeof(double));
  memset(A_, 0, L_a * rankSizeExt * sizeof(double));
  memset(B_, 0, L_b * rankSizeExt * sizeof(double));
  memset(C_, 0, L_c * rankSizeExt * sizeof(double));
  // compute the extension matrices
  for (uint m = 0; m < rankSize; ++m) {
    for (uint n = 0; n < rankSize; ++n) {
      // extension for matrix A
      size_t r = m * rankSize + n;
      double sum = 0;
      for (uint i = 0; i < L_a; ++i) {
        sum += abs(A[m * L_a + i] * A[n * L_a + i]);
        A_[r * L_a + i] = sum;
      }
      // extension for matrix B
      sum = 0;
      for (uint j = 0; j < L_b; ++j) {
        sum += abs(B[m * L_b + j] * B[n * L_b + j]);
        B_[r * L_b + j] = sum;
      }
      // extension for matrix C
      sum = 0;
      for (uint k = 0; k < L_c; ++k) {
        sum += abs(C[m * L_c + k] * C[n * L_c + k]);
        C_[r * L_c + k] = sum;
      }
    }
  }
  // extension matrices
  FactorMat MatA_(L_a, rankSizeExt, A_, MATRIX_NONE_SUM);
  FactorMat MatB_(L_b, rankSizeExt, B_, MATRIX_NONE_SUM);
  FactorMat MatC_(L_c, rankSizeExt, C_, MATRIX_NONE_SUM);
  //-------------------------------------
  // Compute weight
  //-------------------------------------
  double SumofW = 0;
  double *weight = (double *)malloc(rankSizeExt * sizeof(double));
  memset(weight, 0, rankSizeExt * sizeof(double));
  for (uint r = 0; r < rankSizeExt; ++r) {
    weight[r] = MatA_(L_a - 1, r);
    weight[r] *= MatB_(L_b - 1, r);
    weight[r] *= MatC_(L_c - 1, r);
    SumofW += weight[r];
  }
  Time_.r_init(start);
  mexPrintf("|-%f during the initialization phase.\n", Time_.initialization);
  mexEvalString("drawnow");
  //-------------------------
  // Do Sampling
  //-------------------------
  start = clock();
  size_t TotalS = 0;
  size_t *freq_r = (size_t *)malloc(rankSizeExt * sizeof(size_t));
  memset(freq_r, 0, rankSizeExt * sizeof(size_t));
  for (uint r = 0; r < rankSizeExt; ++r) {
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
  uint *IdxK = (uint *)malloc(TotalS * sizeof(uint));
  memset(IdxK, 0, TotalS * sizeof(uint));
  // sample indexes
  size_t offset = 0;
  for (uint r = 0; r < rankSizeExt; ++r) {
    binary_search(freq_r[r], (IdxI + offset), L_a, (MatA_.value + r * L_a));
    binary_search(freq_r[r], (IdxJ + offset), L_b, (MatB_.value + r * L_b));
    binary_search(freq_r[r], (IdxK + offset), L_c, (MatC_.value + r * L_c));
    offset += freq_r[r];
  }
  Time_.r_samp(start);
  mexPrintf("|-%f during the sampling phase.\n", Time_.sampling);
  mexEvalString("drawnow");
  //-----------------------------------------
  // Filtering
  //-----------------------------------------
  start = clock();
  // use map IrJc to save the sampled values
  Point3dValueMap IrJc;
  if (budget >= NumSample) {
    for (size_t i = 0; i < TotalS; ++i) {
      IrJc[Point3d(IdxI[i], IdxJ[i], IdxK[i])] = 1.0;
    }
  } else {
    offset = 0;
    for (uint m = 0; m < rankSize; ++m) {
      for (uint n = 0; n < rankSize; ++n) {
        size_t r = m * rankSize + n;
        for (size_t s = 0; s < freq_r[r]; ++s, ++offset) {
          uint idxi = IdxI[offset];
          uint idxj = IdxJ[offset];
          uint idxk = IdxK[offset];
          int score = sgn(AT(m, idxi)) * sgn(AT(n, idxi));
          score *= sgn(BT(m, idxj)) * sgn(BT(n, idxj));
          score *= sgn(CT(m, idxk)) * sgn(CT(n, idxk));
          IrJc[Point3d(idxi, idxj, idxk)] += (double)score;
        }
      }
    }
  }
  Time_.r_score(start);
  mexPrintf("|-%f during the scoring phase.\n", Time_.scoring);
  mexEvalString("drawnow");
  //-----------------------------------
  // sort the values have been sampled
  //-----------------------------------
  // for pre-sort
  start = clock();
  std::vector<Point3dValuePair> tempSortedVec;
  std::vector<Point3dValuePair> sortVec;
  // push the value into a vector for sorting
  for (auto mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr) {
    tempSortedVec.push_back(std::make_pair(mapItr->first, mapItr->second));
  }
  sort(tempSortedVec.begin(), tempSortedVec.end(), compgt<Point3dValuePair>);
  for (size_t m = 0; m < tempSortedVec.size() && m < budget; ++m) {
    double true_value = MatrixColMul(tempSortedVec[m].first, AT, BT, CT);
    sortVec.push_back(std::make_pair(tempSortedVec[m].first, true_value));
  }
  // sort the vector according to the actual value
  sort(sortVec.begin(), sortVec.end(), compgt<Point3dValuePair>);
  Time_.r_filter(start);
  mexPrintf("|-%f during the sorting phase.\n", Time_.filtering);
  mexEvalString("drawnow");
  //--------------------------------
  // Converting to Matlab
  //--------------------------------
  // value
  for (size_t m = 0; m < sortVec.size() && m < top_t; ++m) {
    // value
    values[m] = sortVec[m].second;
    // indexes
    indexes[m] = (sortVec[m].first.x + 1);
    indexes[m + top_t] = (sortVec[m].first.y + 1);
    indexes[m + top_t + top_t] = (sortVec[m].first.z + 1);
  }
  tsec[0] = Time_.initialization;
  tsec[1] = Time_.sampling;
  tsec[2] = Time_.scoring;
  tsec[3] = Time_.filtering;
  mexPrintf("Done!\n");
  //---------------
  // free
  //---------------
  free(weight);
  free(IdxI);
  free(IdxJ);
  free(IdxK);
  free(freq_r);
  free(A_);
  free(B_);
  free(C_);
}
