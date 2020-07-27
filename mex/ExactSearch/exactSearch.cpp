#include <algorithm>
#include <ctime>
#include <list>
#include <vector>

#include "include/matrix.h"
#include "include/utils.h"
#include "mex.h"

double ColMul(const uint *curIdx, double **p, uint rank, uint numMat) {
  double ans = 0.0;
  for (uint r = 0; r < rank; ++r) {
    double temp = 1.0;
    for (uint i = 0; i < numMat; ++i) {
      temp *= p[i][curIdx[i] * rank + r];
    }
    ans += temp;
  }
  return ans;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  clock_t start, finish;
  //---------------------
  // initialization
  //---------------------
  double duration;
  double total = 1.0;
  double progress = 0.0;
  double progress_flag = 0.0;
  const uint rank = (uint)mxGetM(prhs[0]);
  const uint numMat = nrhs - 1;
  const uint top_t = (uint)mxGetPr(prhs[nrhs - 1])[0];
  uint *max = (uint *)malloc(numMat * sizeof(uint));
  memset(max, 0, numMat * sizeof(uint));
  double **Mats = (double **)malloc(numMat * sizeof(double *));
  for (uint i = 0; i < numMat; ++i) {
    Mats[i] = mxGetPr(prhs[i]);
    max[i] = mxGetN(prhs[i]);
    total *= max[i];
  }
  //------------------------
  // Do exhaustive computing
  //------------------------
  std::list<double> listTop;
  progressbar(progress);
  // subIndex for loop
  start = clock();
  IndexCounter index(numMat, max);
  // compute top_t values as the initial list
  for (uint count = 0; count < top_t && !index.isDone(); ++index, ++count) {
    double tempValue = ColMul(index.getIdx(), Mats, rank, numMat);
    listTop.push_back(tempValue);
    progress += 1;
  }
  // sort the list in descending order
  listTop.sort();
  listTop.reverse();
  // do exhaustive search
  while (!index.isDone()) {
    double tempValue = ColMul(index.getIdx(), Mats, rank, numMat);
    if (tempValue > listTop.back()) {
      doInsert(tempValue, listTop);
    }
    ++index;
    progress += 1;
    if (progress_flag > 1e8) {
      clearprogressbar();
      progressbar(progress / total);
      progress_flag = 0;
    }
  }
  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  //-----------------------------
  // convert the result to Matlab
  //-----------------------------
  plhs[0] = mxCreateDoubleMatrix(listTop.size(), 1, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  mxGetPr(plhs[1])[0] = duration;
  double *topValue = mxGetPr(plhs[0]);
  std::list<double>::iterator itr = listTop.begin();
  for (uint i = 0; i < listTop.size(); ++i) {
    topValue[i] = *itr++;
  }
  clearprogressbar();
  progressbar(1.0);
  //-------------------
  // free
  //--------------------
  free(max);
  free(Mats);
}
