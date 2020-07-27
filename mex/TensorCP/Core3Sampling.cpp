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
  clock_t start;
  srand(unsigned(time(NULL)));
  //--------------------
  // Initialization
  //--------------------
  uint rankSize = (uint)mxGetN(prhs[0]);
  uint rankSizeExt = rankSize * rankSize * rankSize;
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
  mexPrintf("Starting Core^3 Sampling:");
  mexPrintf("Top:%d,Samples:1e%d,Budget:1e%d\n", top_t, (int)log10(NumSample),
            (int)log10(budget));
  mexEvalString("drawnow");
  // compute the extension for matrices
  Timer Time_;
  start = clock();
  // original matrices
  FactorMat MatA(L_a, rankSize, A, MATRIX_NONE_SUM);
  FactorMat MatB(L_b, rankSize, B, MATRIX_NONE_SUM);
  FactorMat MatC(L_c, rankSize, C, MATRIX_NONE_SUM);
  AT.transpose(A);
  BT.transpose(B);
  CT.transpose(C);
  //-------------------------------------
  // Compute weight
  //-------------------------------------
  double SumofW = 0;
  double *weight = (double *)malloc(rankSizeExt * sizeof(double));
  memset(weight, 0, rankSizeExt * sizeof(double));
  for (uint m = 0; m < rankSize; ++m) {
    for (uint n = 0; n < rankSize; ++n) {
      for (uint h = 0; h < rankSize; ++h) {
        size_t r = m * rankSize * rankSize + n * rankSize + h;
        double p = 0.0;
        double q = 0.0;
        double t = 0.0;
        for (uint i = 0; i < L_a; ++i) {
          p += abs(MatA(i, m) * MatA(i, n) * MatA(i, h));
        }
        for (uint j = 0; j < L_b; ++j) {
          q += abs(MatB(j, m) * MatB(j, n) * MatB(j, h));
        }
        for (uint k = 0; k < L_c; ++k) {
          t += abs(MatC(k, m) * MatC(k, n) * MatC(k, h));
        }
        weight[r] = p * q * t;
        SumofW += p * q * t;
      }
    }
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
  double *pa = (double *)malloc(L_a * sizeof(double));
  double *pb = (double *)malloc(L_b * sizeof(double));
  double *pc = (double *)malloc(L_c * sizeof(double));
  memset(pa, 0, L_a * sizeof(double));
  memset(pb, 0, L_b * sizeof(double));
  memset(pc, 0, L_c * sizeof(double));
  size_t offset = 0;
  for (uint m = 0; m < rankSize; ++m) {
    for (uint n = 0; n < rankSize; ++n) {
      for (uint h = 0; h < rankSize; ++h) {
        size_t r = m * rankSize * rankSize + n * rankSize + h;
        double sum_a = 0.0;
        double sum_b = 0.0;
        double sum_c = 0.0;
        for (uint i = 0; i < L_a; ++i) {
          sum_a += abs(MatA(i, m) * MatA(i, n) * MatA(i, h));
          pa[i] = sum_a;
        }
        for (uint j = 0; j < L_b; ++j) {
          sum_b += abs(MatB(j, m) * MatB(j, n) * MatB(j, h));
          pb[j] = sum_b;
        }
        // extension for matrix C
        for (uint k = 0; k < L_c; ++k) {
          sum_c += abs(MatC(k, m) * MatC(k, n) * MatC(k, h));
          pc[k] = sum_c;
        }
        binary_search(freq_r[r], (IdxI + offset), L_a, pa);
        binary_search(freq_r[r], (IdxJ + offset), L_b, pb);
        binary_search(freq_r[r], (IdxK + offset), L_c, pc);
        offset += freq_r[r];
      }
    }
  }
  Time_.r_samp(start);
  mexPrintf("|-%f during the sampling phase.\n", Time_.sampling);
  mexEvalString("drawnow");
  //-----------------------------------------
  // Filtering
  //-----------------------------------------
  start = clock();
  Point3dValueMap IrJc;
  if (budget >= NumSample) {
    for (size_t i = 0; i < TotalS; ++i) {
      IrJc[Point3d(IdxI[i], IdxJ[i], IdxK[i])] = 1.0;
    }
  } else {
    offset = 0;
    for (uint m = 0; m < rankSize; ++m) {
      for (uint n = 0; n < rankSize; ++n) {
        for (uint h = 0; h < rankSize; ++h) {
          size_t r = m * rankSize * rankSize + n * rankSize + h;
          for (size_t s = 0; s < freq_r[r]; ++s) {
            uint idxi = IdxI[offset];
            uint idxj = IdxJ[offset];
            uint idxk = IdxK[offset];
            int score =
                sgn(MatA(idxi, m)) * sgn(MatA(idxi, n)) * sgn(MatA(idxi, h));
            score *=
                sgn(MatB(idxj, m)) * sgn(MatB(idxj, n)) * sgn(MatB(idxj, h));
            score *=
                sgn(MatC(idxk, m)) * sgn(MatC(idxk, n)) * sgn(MatC(idxk, h));
            IrJc[Point3d(idxi, idxj, idxk)] += (double)score;
            ++offset;
          }
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
  free(pa);
  free(pb);
  free(pc);
}
