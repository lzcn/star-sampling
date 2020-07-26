#include <ctime>
#include <list>

#include "matrix.h"
#include "mex.h"
#include "utils.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  clock_t start, finish;
  const uint NumQueries = mxGetN(prhs[0]);
  // factor matrices
  FactorMat A(mxGetM(prhs[0]), mxGetN(prhs[0]), mxGetPr(prhs[0]));
  FactorMat B(mxGetM(prhs[1]), mxGetN(prhs[1]), mxGetPr(prhs[1]));
  FactorMat C(mxGetM(prhs[2]), mxGetN(prhs[2]), mxGetPr(prhs[2]));
  const uint knn = mxGetPr(prhs[3])[0];
  const uint NumMat = 2;
  // result values for each query
  plhs[0] = mxCreateDoubleMatrix(knn, NumQueries, mxREAL);
  double *knnValue = mxGetPr(plhs[0]);
  // sampling time for each query
  plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  double *duration = mxGetPr(plhs[1]);
  //---------------------
  // Start full search
  //---------------------
  mexPrintf("Starting Exhaustive Search for Queries......\n");
  mexEvalString("drawnow");
  uint *maxIdx = (uint *)malloc(NumMat * sizeof(uint));
  maxIdx[0] = mxGetN(prhs[1]);
  maxIdx[1] = mxGetN(prhs[2]);
  IndexCounter index(NumMat, maxIdx);
  progressbar(0);
  start = clock();
  for (uint i = 0; i < NumQueries; ++i) {
    clearprogressbar();
    progressbar((double)i / NumQueries);
    index.reset();
    std::list<double> listTop;
    for (uint count = 0; count < knn && !index.isDone(); ++index) {
      Point3d p(i, index.getIdx()[0], index.getIdx()[1]);
      double temp = MatrixColMul(p, A, B, C);
      listTop.push_back(temp);
      ++count;
    }
    listTop.sort();
    listTop.reverse();
    while (!index.isDone()) {
      Point3d p(i, index.getIdx()[0], index.getIdx()[1]);
      double temp = MatrixColMul(p, A, B, C);
      if (temp > listTop.back()) {
        doInsert(temp, listTop);
      }
      ++index;
    }
    std::list<double>::iterator itr = listTop.begin();
    for (uint p = 0; p < knn && p < listTop.size(); ++p) {
      knnValue[i * knn + p] = *itr++;
    }
  }
  finish = clock();
  duration[0] = (double)(finish - start) / (NumQueries * CLOCKS_PER_SEC);
  clearprogressbar();
  progressbar(1);
  mexPrintf("\n");
  free(maxIdx);
}
