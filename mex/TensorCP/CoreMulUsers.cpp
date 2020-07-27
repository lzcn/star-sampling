/*
        Central Sampling for queries
        [value, time] = queryCentralSampling(A, B, C, budget, samples, knn)
                A: size(L1, R)
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
  // normal matrix
  double *A = mxGetPr(prhs[0]);
  double *B = mxGetPr(prhs[1]);
  double *C = mxGetPr(prhs[2]);
  uint L_a = (uint)mxGetM(prhs[0]);
  uint L_b = (uint)mxGetM(prhs[1]);
  uint L_c = (uint)mxGetM(prhs[2]);
  // number of queries
  const uint NumQueries = (uint)mxGetM(prhs[0]);
  // rank size
  const uint rankSize = (uint)mxGetN(prhs[0]);
  // MatA is a set of queries

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
  mexPrintf("Core^2 Sampling for MultiUsers:");
  mexPrintf("Top:%d,Samples:1e%d,Budget:1e%d,Number of Queries:%d\n", knn,
            (int)log10(NumSample), (int)log10(budget), NumQueries);
  mexEvalString("drawnow");
  progressbar(0);
  start = clock();
  FactorMat MatA(L_a, rankSize);
  FactorMat MatB(L_b, rankSize);
  FactorMat MatC(L_c, rankSize);
  MatA.accumulation(A);
  MatB.accumulation(B);
  MatC.accumulation(C);
  FactorMat AT(rankSize, L_a);
  FactorMat BT(rankSize, L_b);
  FactorMat CT(rankSize, L_c);
  AT.transpose(A);
  BT.transpose(B);
  CT.transpose(C);
  finish = clock();
  for (uint i = 0; i < NumQueries; i++) {
    SamplingTime[i] = (double)(finish - start) / (CLOCKS_PER_SEC * NumQueries);
  }
  //-------------------------------------
  // Compute weight
  //-------------------------------------
  // weight for each query
  double *weight = (double *)malloc(rankSize * sizeof(double));
  memset(weight, 0, rankSize * sizeof(double));
  size_t *freq_r = (size_t *)malloc(rankSize * sizeof(size_t));
  memset(freq_r, 0, rankSize * sizeof(size_t));
  uint *IdxJ = (uint *)malloc(NumSample * sizeof(uint));
  uint *IdxK = (uint *)malloc(NumSample * sizeof(uint));
  memset(IdxJ, 0, NumSample * sizeof(uint));
  memset(IdxK, 0, NumSample * sizeof(uint));
  //-------------------------
  // Do Sampling
  //-------------------------
  // list for sub walk
  std::vector<std::vector<Point2d> > subWalk(rankSize);
  double SumofW = 0.0;
  for (uint i = 0; i < NumQueries; ++i) {
    clearprogressbar();
    progressbar((double)i / NumQueries);
    start = clock();
    SumofW = 0.0;
    for (uint r = 0; r < rankSize; ++r) {
      weight[r] = abs(MatA(i, r));
      weight[r] *= MatB(L_b - 1, r);
      weight[r] *= MatC(L_c - 1, r);
      SumofW += weight[r];
    }
    finish = clock();
    SamplingTime[i] += (double)(finish - start) / CLOCKS_PER_SEC;
    if (SumofW == 0) {
      continue;
    }
    start = clock();
    for (uint r = 0; r < rankSize; ++r) {
      double u = (double)rand() / (double)RAND_MAX;
      double c = (double)NumSample * weight[r] / SumofW;
      if (u < (c - floor(c)))
        freq_r[r] = (size_t)ceil(c);
      else
        freq_r[r] = (size_t)floor(c);
    }
    Point3dValueMap IrJc;
    for (uint r = 0; r < rankSize; ++r) {
      // Check the list length for each query
      if (freq_r[r] > subWalk[r].size()) {
        size_t remain = freq_r[r] - subWalk[r].size();
        binary_search(remain, IdxJ, L_b, (MatB.value + r * L_b));
        binary_search(remain, IdxK, L_c, (MatC.value + r * L_c));
        for (uint p = 0; p < remain; ++p) {
          subWalk[r].push_back(Point2d(IdxJ[p], IdxK[p]));
        }
      }
      for (size_t m = 0; m < freq_r[r]; ++m) {
        // repeat c_r times to sample indexes j, k
        uint idxJ = (subWalk[r])[m].x;
        uint idxK = (subWalk[r])[m].y;
        IrJc[Point3d(i, idxJ, idxK)] +=
            sgn(MatA(i, r)) * sgn(MatB(idxJ, r)) * sgn(MatC(idxK, r));
      }
    }
    finish = clock();
    SamplingTime[i] += (double)(finish - start) / CLOCKS_PER_SEC;
    //-----------------------------------
    // sort the values have been sampled
    //-----------------------------------
    std::vector<Point3dValuePair> tempSortedVec;
    std::vector<Point3dValuePair> sortVec;
    for (auto mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr) {
      tempSortedVec.push_back(std::make_pair(mapItr->first, mapItr->second));
    }
    start = clock();
    sort(tempSortedVec.begin(), tempSortedVec.end(), compgt<Point3dValuePair>);
    for (uint t = 0; t < tempSortedVec.size() && t < budget; ++t) {
      double true_value = MatrixColMul(tempSortedVec[t].first, AT, BT, CT);
      sortVec.push_back(std::make_pair(tempSortedVec[t].first, true_value));
    }
    sort(sortVec.begin(), sortVec.end(), compgt<Point3dValuePair>);
    finish = clock();
    SamplingTime[i] += (double)(finish - start) / CLOCKS_PER_SEC;
    for (uint s = 0; s < sortVec.size() && s < knn; ++s) {
      knnValue[i * knn + s] = sortVec[s].second;
    }
  }
  clearprogressbar();
  progressbar(1);
  mexPrintf("\n");
  //---------------
  // free
  //---------------
  free(weight);
  free(freq_r);
  free(IdxJ);
  free(IdxK);
}
