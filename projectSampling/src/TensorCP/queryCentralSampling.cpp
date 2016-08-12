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

#include <vector>
#include <map>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <ctime>

#include "mex.h"
#include "matrix.h"

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	clock_t start,finish;
	double duration;
	srand(unsigned(time(NULL)));
	//--------------------
	// Initialization
	//--------------------
	// number of queries
	const size_t NumQueries = mxGetM(prhs[0]);
	// rank size
	const size_t rankSize = mxGetN(prhs[0]);
	// MatA is a set of queries
	start = clock();
	Matrix MatA(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
	Matrix MatB(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
	Matrix MatC(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));
	finish = clock();
	// budget
	const size_t budget = (size_t)mxGetPr(prhs[3])[0];
	// sample number
	const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
	// kNN
	const size_t knn = (size_t)mxGetPr(prhs[5])[0];
	// result values for each query
	plhs[0] = mxCreateDoubleMatrix(knn, NumQueries, mxREAL);
	double *knnValue = mxGetPr(plhs[0]);
	// sampling time for each query
	plhs[1] = mxCreateDoubleMatrix(NumQueries, 1, mxREAL);	
	double *SamplingTime = mxGetPr(plhs[1]);
	memset(SamplingTime, 0, NumQueries*sizeof(double));
	for (size_t i = 0; i < NumQueries; ++i) {
		SamplingTime[i] = (double)(finish-start)/NumQueries;
	}
	mexPrintf("Central Sampling for Queries: samples - %7.0f, budget - %7.0f, knn - %4.0f", NumSample, budget, knn);
	//-------------------------------------
	// Compute weight
	//-------------------------------------
	// weight for each query
	double *weight = (double*)malloc(NumQueries*rankSize*sizeof(double));
	memset(weight, 0,NumQueries*rankSize*sizeof(double));
	// sum of weight for each query
	double *SumofW = (double*)malloc(NumQueries*sizeof(double));
	memset(SumofW, 0, NumQueries*sizeof(double));
	// is the query all zeros
	int *isZero = (int*)malloc(NumQueries*sizeof(int));
	memset(isZero, 0, NumQueries *sizeof(int));	
	for(size_t i = 0; i < NumQueries; ++i){
		start = clock();
		for (size_t r = 0; r < rankSize; ++r){
			//each query's weight is q'_r = |q_r|*||b_{*r}||_1||c_{*r}||_1
			double tempW;
			tempW = abs(MatA.GetElement(i,r));
			tempW *= MatB.SumofCol[r];
			tempW *= MatC.SumofCol[r];
			weight[i*rankSize + r] = tempW;
			SumofW[i] += tempW;
		}
		if(SumofW[i] == 0)
			isZero[i] = 1;
		finish = clock();
		SamplingTime[i] += (double)(finish-start);
	}
	//-------------------------------
	// Compute c_r for each query
	//-------------------------------
	size_t *freq_r = (size_t*)malloc(NumQueries*rankSize*sizeof(size_t));
	memset(freq_r, 0, NumQueries*rankSize*sizeof(size_t));
	// freq_r[r] has the expectation (NumSample*w_r)/|w|_1
	for(size_t i = 0; i < NumQueries; ++i){
		// if is all zero query skip it
		if(isZero[i] == 1){
			continue;
		}
		start = clock();
		for (size_t r = 0; r < rankSize; ++r){
			double u = (double)rand()/(double)RAND_MAX;
			double c = (double)NumSample*weight[i*rankSize + r]/SumofW[i];
			if(u < (c - floor(c)))
				freq_r[i*rankSize + r] = (size_t)ceil(c);
			else
				freq_r[i*rankSize + r] = (size_t)floor(c);
		}
		finish = clock();
		SamplingTime[i] += (double)(finish-start);
	}
	//-------------------------
	// Do Sampling
	//-------------------------
	// list for sub walk
	std::vector<std::vector<point2D>> subWalk(rankSize);
	for(size_t i = 0; i < NumQueries; ++i){
		if(isZero[i] == 1){
			continue;
		}
		start = clock();
		std::map<point3D, double> IrJc;
		for(size_t r = 0; r < rankSize; ++r){
			// Check the list length for each query
			if(freq_r[i*rankSize + r] > subWalk[r].size()){
				size_t remain = freq_r[i*rankSize + r] - subWalk[r].size();
				size_t *IdxJ = (size_t*)malloc(remain*sizeof(size_t));
				size_t *IdxK = (size_t*)malloc(remain*sizeof(size_t));
				memset(IdxJ, 0, remain*sizeof(size_t));
				memset(IdxK, 0, remain*sizeof(size_t));
				vose_alias(remain, IdxJ, \
					MatB.row, \
					(MatB.element + r*MatB.row), \
					MatB.SumofCol[r]);
				vose_alias(remain, IdxK, \
					MatC.row, \
					(MatC.element + r*MatC.row), \
					MatC.SumofCol[r]);
				for(size_t p = 0; p < remain; ++p){
					subWalk[r].push_back(point2D(IdxJ[p],IdxK[p]));
				}
				free(IdxJ);
				free(IdxK);
			}
			for(size_t m = 0; m < freq_r[i*rankSize + r]; ++m){
				// repeat c_r times to sample indexes j, k
				size_t idxJ = (subWalk[r])[m].x;
				size_t idxK = (subWalk[r])[m].y;
				double valueSampled;
				valueSampled = sgn_foo(MatA.GetElement(i,r));
				valueSampled *= sgn_foo(MatB.GetElement(idxJ,r));
				valueSampled *= sgn_foo(MatC.GetElement(idxK,r));
				IrJc[point3D(i,idxJ,idxK)] += valueSampled;
			}
		}
		// compute the score for each query
		std::vector<pidx3d> tempSortedVec;
		for (auto mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr) {
			tempSortedVec.push_back(std::make_pair(mapItr->first, mapItr->second));
		}
		sort(tempSortedVec.begin(), tempSortedVec.end(), compgt<pidx3d>);
		std::vector<pidx3d> sortVec;
		for(size_t t = 0; t < tempSortedVec.size() && t < budget; ++t){
			double true_value = MatrixColMul(tempSortedVec[t].first, MatA, MatB, MatC);
			sortVec.push_back(std::make_pair(tempSortedVec[t].first,true_value));
		}
		sort(sortVec.begin(),sortVec.end(),compgt<pidx3d>);
		finish = clock();
		SamplingTime[i] += (double)(finish-start);
		SamplingTime[i] /= CLOCKS_PER_SEC;
		for(size_t s = 0; s < sortVec.size() && s < knn; ++s){
			knnValue[i*knn + s] = sortVec[s].second;
		}
	}
	//---------------
	// free
	//---------------
	free(weight);
	free(isZero);
	free(freq_r);
	free(SumofW);
}
