/*
        Diamond Sampling for matrix multiplication
        Author: Zhi Lu
        Reference:"Diamond Sampling for Approximate Maximum
                        All-pairs Dot-product(MAD) Search"
*/
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <map>
#include <vector>

#include "matrix.h"
#include "mex.h"
#include "utils.h"
/*
        diamond sampling for matrix
        A's size: Rank x M
        B's size: N x Rank
        [value, time] = diamondMatrix(A,B,budget,samples,top_t);
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  clock_t start, finish;
  double duration;
  srand(unsigned(time(NULL)));
  //--------------------
  // Initialization
  //--------------------
  start = clock();
  FactorMat MatA(mxGetM(prhs[0]), mxGetN(prhs[0]), mxGetPr(prhs[0]),
              MATRIX_COL_SUM);
  FactorMat MatB(mxGetM(prhs[1]), mxGetN(prhs[1]), mxGetPr(prhs[1]),
              MATRIX_COL_SUM);
  FactorMat BT(mxGetN(prhs[1]), mxGetM(prhs[1]));
  uint rankSize = mxGetM(prhs[0]);
  for (uint r = 0; r < rankSize; ++r) {
    for (uint j = 0; j < MatB.n_row; ++j) {
      BT(r, j) = MatB(j, r);
    }
  }
  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  // get the budget
  const size_t budget = (size_t)mxGetPr(prhs[2])[0];
  // get the  number of samples
  const size_t NumSample = (size_t)mxGetPr(prhs[3])[0];
  // get the top_t
  const uint top_t = (uint)mxGetPr(prhs[4])[0];
  // value
  plhs[0] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
  double *plhs_result = mxGetPr(plhs[0]);
  // result for time
  plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  double *tsec = mxGetPr(plhs[1]);
  *tsec = duration;
  // pair
  plhs[2] = mxCreateNumericMatrix(top_t, 2, mxUINT64_CLASS, mxREAL);
  uint64_T *plhs_pr = (uint64_T *)mxGetData(plhs[2]);
  mexPrintf("Starting Diamond Sampling:");
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
      double tempW = abs(MatA(r, i)) * MatA.col_abs_sum[i] * MatB.col_abs_sum[r];
      weight[r * MatA.n_col + i] = tempW;
      SumofW += tempW;
    }
  }
  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  *tsec += duration;
  //-------------------------
  // Do Sampling
  //-------------------------
  start = clock();
  // sampled r, i, j, r'
  uint *IdxR = (uint *)malloc(NumSample * sizeof(uint));
  memset(IdxR, 0, NumSample * sizeof(uint));
  uint *IdxI = (uint *)malloc(NumSample * sizeof(uint));
  memset(IdxI, 0, NumSample * sizeof(uint));
  uint *IdxJ = (uint *)malloc(NumSample * sizeof(uint));
  memset(IdxJ, 0, NumSample * sizeof(uint));
  uint *IdxRp = (uint *)malloc(NumSample * sizeof(uint));
  memset(IdxRp, 0, NumSample * sizeof(uint));
  // sampled r's frequency
  size_t *freq_r = (size_t *)malloc(MatA.n_row * sizeof(size_t));
  memset(freq_r, 0, MatA.n_row * sizeof(size_t));
  double *pdfb = (double *)malloc(MatB.n_row * sizeof(double));
  // sample pairs (i,r) ,
  sort_sample(NumSample, IdxI, IdxR, freq_r, MatA.n_row, MatA.n_col, weight,
              SumofW);
  // sample j;
  for (uint r = 0, offset = 0; r < MatA.n_row; ++r) {
    double sum = 0;
    for (uint i = 0; i < MatB.n_row; ++i) {
      sum += abs(MatB(i, r));
      pdfb[i] = sum;
    }
    binary_search(freq_r[r], (IdxJ + offset), MatB.n_row, pdfb);
    // vose_alias( freq_r[r], (IdxJ + offset), MatB.row, (MatB.element +
    // r*MatB.row), MatB.SumofCol[r]);
    offset += freq_r[r];
  }
  // sample rp and  get score
  Point2dValueMap IrJc;
  for (size_t s = 0; s < NumSample; ++s) {
    uint r = IdxR[s];
    uint i = IdxI[s];
    uint j = IdxJ[s];
    uint rp = MatA.randRow(i);
    // Update the element in coordinate
    IrJc[Point2d(i, j)] +=
        sgn(MatA(r, i)) * sgn(MatB(j, r)) * sgn(MatA(rp, i)) * MatB(j, rp);
  }
  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  *tsec += duration;

  //-----------------------------------
  // sort the values have been sampled
  //-----------------------------------
  std::vector<Point2dValuePair> sortVec;
  std::vector<Point2dValuePair> tempSortedVec;
  // sort the sampled value
  for (auto mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr) {
    tempSortedVec.push_back(std::make_pair(mapItr->first, mapItr->second));
  }
  start = clock();
  sort(tempSortedVec.begin(), tempSortedVec.end(), compgt<Point2dValuePair>);
  finish = clock();
  *tsec += duration;
  // compute the actual of top-t'(budget)
  for (size_t m = 0; m < tempSortedVec.size() && m < budget; ++m) {
    double true_value = vectors_mul(tempSortedVec[m].first, MatA, MatB);
    sortVec.push_back(std::make_pair(tempSortedVec[m].first, true_value));
  }
  sort(sortVec.begin(), sortVec.end(), compgt<Point2dValuePair>);
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
  }
  mexPrintf("Done!\n");
  //---------------
  // free
  //---------------
  free(weight);
  free(IdxI);
  free(IdxJ);
  free(IdxR);
  free(IdxRp);
  free(freq_r);
  free(pdfb);
}
