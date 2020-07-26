/*
        Diamond Sampling for queries
        [value, time] = querySampling(A, B, C, budget, samples, knn)
                A: size(R, L1)
                B: size(L2, R)
                C: size(L3, R)
        output:
                value: size(knn, NumQueries)
                time: size(NumQueries, 1)

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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  clock_t start, finish;
  double duration;
  srand(unsigned(time(NULL)));
  //--------------------
  // Initialization
  //--------------------
  // sampling time for each query
  const uint NumQueries = mxGetN(prhs[0]);
  const uint rankSize = mxGetM(prhs[0]);
  // MatA is a set of queries
  FactorMat MatA(mxGetM(prhs[0]), mxGetN(prhs[0]), mxGetPr(prhs[0]));
  FactorMat MatB(mxGetM(prhs[1]), mxGetN(prhs[1]), mxGetPr(prhs[1]));
  FactorMat MatC(mxGetM(prhs[2]), mxGetN(prhs[2]), mxGetPr(prhs[2]));
  // budget
  const size_t budget = (size_t)mxGetPr(prhs[3])[0];
  // sample number
  const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
  // kNN
  const uint knn = (uint)mxGetPr(prhs[5])[0];

  // result values for each query
  plhs[0] = mxCreateDoubleMatrix(knn, NumQueries, mxREAL);
  double *knnValue = mxGetPr(plhs[0]);
  // sampling time for each query
  plhs[1] = mxCreateDoubleMatrix(NumQueries, 1, mxREAL);
  double *SamplingTime = mxGetPr(plhs[1]);
  memset(SamplingTime, 0, NumQueries * sizeof(double));
  mexPrintf("Diamond Sampling for Queries");
  mexPrintf("- Top-%d ", knn);
  mexPrintf("- Samples:1e%d ", (int)log10(NumSample));
  mexPrintf("- Budget:1e%d ", (int)log10(budget));
  mexPrintf("......");
  mexEvalString("drawnow");
  //-------------------------------------
  // Compute weight
  //-------------------------------------
  double *weight = (double *)malloc(rankSize * sizeof(double));
  memset(weight, 0, rankSize * sizeof(double));
  size_t *freq_r = (size_t *)malloc(rankSize * sizeof(size_t));
  memset(freq_r, 0, rankSize * sizeof(size_t));
  uint *idxRp = (uint *)malloc((NumSample + rankSize) * sizeof(uint));
  memset(idxRp, 0, (NumSample + rankSize) * sizeof(uint));
  //-------------------------
  // Do Sampling
  //-------------------------
  // list pool for sub walk
  std::vector<std::vector<Point2d> > subWalk(MatA.n_row);
  for (uint i = 0; i < NumQueries; ++i) {
    double SumofW = 0.0;
    start = clock();
    for (uint r = 0; r < rankSize; ++r) {
      weight[r] = abs(MatA.GetElement(r, i));
      weight[r] *= MatB.col_abs_sum[r];
      weight[r] *= MatC.col_abs_sum[r];
      SumofW += weight[r];
    }
    finish = clock();
    SamplingTime[i] += (double)(finish - start);
    if (SumofW == 0) continue;
    // compute c[r]
    start = clock();
    for (uint r = 0; r < rankSize; ++r) {
      double u = (double)rand() / (double)RAND_MAX;
      double c = (double)NumSample * weight[r] / SumofW;
      if (u < (c - floor(c)))
        freq_r[r] = (size_t)ceil(c);
      else
        freq_r[r] = (size_t)floor(c);
    }
    finish = clock();
    SamplingTime[i] += (double)(finish - start);
    // sample r' for this query
    memset(idxRp, 0, (NumSample + rankSize) * sizeof(uint));
    start = clock();
    vose_alias((NumSample + rankSize), idxRp, rankSize,
               (MatA.value + i * MatA.n_row), MatA.col_abs_sum[i]);
    // use map IrJc to save the sampled values
    std::map<Point3d, double> IrJc;
    // save the sampled values
    for (uint r = 0, offset = 0; r < rankSize; ++r) {
      // Check the list length for each query
      if (freq_r[r] > subWalk[r].size()) {
        size_t remain = freq_r[r] - subWalk[r].size();
        uint *IdxJ = (uint *)malloc(remain * sizeof(uint));
        uint *IdxK = (uint *)malloc(remain * sizeof(uint));
        memset(IdxJ, 0, remain * sizeof(uint));
        memset(IdxK, 0, remain * sizeof(uint));
        vose_alias(remain, IdxJ, MatB.n_row, (MatB.value + r * MatB.n_row),
                   MatB.col_abs_sum[r]);
        vose_alias(remain, IdxK, MatC.n_row, (MatC.value + r * MatC.n_row),
                   MatC.col_abs_sum[r]);
        for (int p = 0; p < remain; ++p) {
          subWalk[r].push_back(Point2d(IdxJ[p], IdxK[p]));
        }
        free(IdxJ);
        free(IdxK);
      }
      // use the pool of indexes to compute the sampled value
      for (size_t m = 0; m < freq_r[r]; ++m) {
        uint rp = idxRp[offset++];
        uint idxJ = (subWalk[r])[m].x;
        uint idxK = (subWalk[r])[m].y;
        double valueSampled = sgn_foo(MatA.GetElement(r, i));
        valueSampled *= sgn_foo(MatB.GetElement(idxJ, r));
        valueSampled *= sgn_foo(MatC.GetElement(idxK, r));
        valueSampled *= sgn_foo(MatA.GetElement(rp, i));
        valueSampled *= MatB.GetElement(idxJ, rp);
        valueSampled *= MatC.GetElement(idxK, rp);
        IrJc[Point3d(i, idxJ, idxK)] += valueSampled;
      }
    }
    finish = clock();
    SamplingTime[i] += (double)(finish - start);
    // pre-sorting the scores
    std::vector<Point3dValuePair> sortVec;
    std::vector<Point3dValuePair> tempSortedVec;
    for (auto mapItr = IrJc.begin(); mapItr != IrJc.end(); mapItr++) {
      tempSortedVec.push_back(std::make_pair(mapItr->first, mapItr->second));
    }
    start = clock();
    sort(tempSortedVec.begin(), tempSortedVec.end(), compgt<Point3dValuePair>);
    // compute the actual value for top-t' indexes
    for (uint t = 0; t < tempSortedVec.size() && t < budget; ++t) {
      double true_value = vectors_mul(tempSortedVec[t].first, MatA, MatB, MatC);
      sortVec.push_back(std::make_pair(tempSortedVec[t].first, true_value));
    }
    sort(sortVec.begin(), sortVec.end(), compgt<Point3dValuePair>);
    finish = clock();
    SamplingTime[i] += (double)(finish - start);
    SamplingTime[i] /= CLOCKS_PER_SEC;
    for (uint s = 0; s < sortVec.size() && s < knn; ++s) {
      knnValue[i * knn + s] = sortVec[s].second;
    }
  }
  mexPrintf("Done!\n");
  //---------------
  // free
  //---------------
  free(weight);
  free(freq_r);
  free(idxRp);
}
