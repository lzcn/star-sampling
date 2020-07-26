#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
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
  // input
  uint rankSize = (uint)mxGetM(prhs[0]);
  uint Acol = (uint)mxGetN(prhs[0]);
  uint L_b = (uint)mxGetM(prhs[1]);
  uint L_c = (uint)mxGetM(prhs[2]);
  double *A = mxGetPr(prhs[0]);
  double *B = mxGetPr(prhs[1]);
  double *C = mxGetPr(prhs[2]);
  Timer Time_;
  start = clock();
  FactorMat MatA(rankSize, Acol);
  FactorMat MatB(L_b, rankSize);
  FactorMat MatC(L_c, rankSize);
  MatA.accumulation(A);
  MatB.accumulation(B);
  MatC.accumulation(C);
  FactorMat AT(rankSize, Acol, A, MATRIX_NONE_SUM);
  FactorMat BT(rankSize, L_b);
  FactorMat CT(rankSize, L_c);
  BT.transpose(B);
  CT.transpose(C);
  Time_.r_init(start);
  const size_t budget = (size_t)mxGetPr(prhs[3])[0];
  const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
  const uint top_t = (uint)mxGetPr(prhs[5])[0];
  // output
  plhs[0] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
  double *values = mxGetPr(plhs[0]);
  // time duration sampling
  plhs[1] = mxCreateDoubleMatrix(1, 4, mxREAL);
  double *tsec = mxGetPr(plhs[1]);
  // indexes of values
  plhs[2] = mxCreateNumericMatrix(top_t, 3, mxUINT64_CLASS, mxREAL);
  uint64_T *indexes = (uint64_T *)mxGetData(plhs[2]);
  mexPrintf("Starting Dimaond Sampling:");
  mexPrintf("Top:%d,Samples:1e%d,Budget:1e%d\n", top_t, (int)log10(NumSample),
            (int)log10(budget));
  mexEvalString("drawnow");
  //-------------------------------------
  // Compute weight
  //-------------------------------------
  double SumofW = 0;
  // weight has the same size of A
  double *weight = (double *)malloc(rankSize * Acol * sizeof(double));
  memset(weight, 0, rankSize * Acol * sizeof(double));
  start = clock();
  for (uint r = 0; r < rankSize; ++r) {
    for (uint i = 0; i < Acol; ++i) {
      double tempW = abs(A[i * rankSize + r]);
      tempW *= MatA(rankSize - 1, i);
      tempW *= MatB(L_b - 1, r);
      tempW *= MatC(L_c - 1, r);
      weight[r * Acol + i] = tempW;
      SumofW += tempW;
    }
  }
  Time_.r_init(start);
  mexPrintf("|-%f during the initialization phase.\n", Time_.initialization);
  mexEvalString("drawnow");
  //-------------------------
  // Do Sampling
  //-------------------------
  // sampled r, i, j, k
  start = clock();
  uint *IdxI = (uint *)malloc(NumSample * sizeof(uint));
  memset(IdxI, 0, NumSample * sizeof(uint));
  uint *IdxJ = (uint *)malloc(NumSample * sizeof(uint));
  memset(IdxJ, 0, NumSample * sizeof(uint));
  uint *IdxK = (uint *)malloc(NumSample * sizeof(uint));
  memset(IdxK, 0, NumSample * sizeof(uint));
  uint *IdxR = (uint *)malloc(NumSample * sizeof(uint));
  memset(IdxR, 0, NumSample * sizeof(uint));
  // sampled r's frequency
  size_t *freq_r = (size_t *)malloc(rankSize * sizeof(size_t));
  memset(freq_r, 0, rankSize * sizeof(size_t));
  // Do sample S pairs (r, i)
  sort_sample(NumSample, IdxI, IdxR, freq_r, rankSize, Acol, weight, SumofW);
  // sample j,k;
  size_t offset = 0;
  for (uint r = 0; r < rankSize; ++r) {
    binary_search(freq_r[r], (IdxJ + offset), MatB.n_row,
                  (MatB.value + r * MatB.n_row));
    binary_search(freq_r[r], (IdxK + offset), MatC.n_row,
                  (MatC.value + r * MatC.n_row));
    offset += freq_r[r];
  }
  Time_.r_samp(start);
  mexPrintf("|-%f during the sampling phase.\n", Time_.sampling);
  mexEvalString("drawnow");
  Point3dValueMap IrJc;
  for (size_t s = 0; s < NumSample; ++s) {
    uint i = IdxI[s];
    uint j = IdxJ[s];
    uint k = IdxK[s];
    uint r = IdxR[s];
    double u = MatA(rankSize - 1, i) * ((double)rand() / (double)RAND_MAX);
    uint rp = binary_search_once(MatA.value + i * rankSize, rankSize - 1, u);
    // Update the element in coordinate
    IrJc[Point3d(i, j, k)] += sgn(AT(r, i)) * sgn(BT(r, j)) * sgn(CT(r, k)) *
                              sgn(AT(rp, i)) * BT(rp, j) * CT(rp, k);
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
  free(IdxR);
  free(freq_r);
}
