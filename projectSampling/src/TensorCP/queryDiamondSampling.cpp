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
	// sampling time for each query
	const size_t NumQueries = mxGetN(prhs[0]);
	const size_t rankSize = mxGetM(prhs[0]);
	// MatA is a set of queries
	Matrix MatA(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
	Matrix MatB(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
	Matrix MatC(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));
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
	mexPrintf("Diamond Sampling for Queries");
	mexPrintf("- Top-%d ",knn);
	mexPrintf("- Samples:1e%d ",(int)log10(NumSample));
	mexPrintf("- Budget:1e%d ",(int)log10(budget));
	mexPrintf("......");
	//-------------------------------------
	// Compute weight
	//-------------------------------------
	double *weight = (double*)malloc(rankSize*sizeof(double));
	memset(weight, 0, rankSize*sizeof(double));
	size_t *freq_r = (size_t*)malloc(rankSize*sizeof(size_t));
	memset(freq_r, 0, rankSize*sizeof(size_t));
	size_t *idxRp = (size_t*)malloc((NumSample + rankSize)*sizeof(size_t));
	memset(idxRp, 0, (NumSample + rankSize)*sizeof(size_t));
	//-------------------------
	// Do Sampling
	//-------------------------
	// list pool for sub walk
	std::vector<std::vector<point2D> > subWalk(MatA.row);
	for(size_t i = 0; i < NumQueries; ++i){
		double SumofW = 0.0;
		start = clock();
		for (size_t r = 0; r < rankSize; ++r){
			weight[r] = abs(MatA.GetElement(r,i));
			weight[r] *= MatB.SumofCol[r];
			weight[r] *= MatC.SumofCol[r];
			SumofW += weight[r];
		}
		finish = clock();
		SamplingTime[i] += (double)(finish-start);
		if(SumofW == 0)
			continue;
		// compute c[r]
		start = clock();
		for (size_t r = 0; r < rankSize; ++r){
			double u = (double)rand()/(double)RAND_MAX;
			double c = (double)NumSample*weight[r]/SumofW;
			if(u < (c - floor(c)))
				freq_r[r] = (size_t)ceil(c);
			else
				freq_r[r] = (size_t)floor(c);
		}
		finish = clock();
		SamplingTime[i] += (double)(finish-start);
		// sample r' for this query
		memset(idxRp, 0, (NumSample + rankSize)*sizeof(size_t));
		start = clock();
		vose_alias((NumSample + rankSize), idxRp, \
					rankSize, \
					(MatA.element + i*MatA.row), \
					MatA.SumofCol[i]);
		// use map IrJc to save the sampled values
		std::map<point3D, double> IrJc;
		// save the sampled values
		for(size_t r = 0,offset = 0; r < rankSize; ++r){
			// Check the list length for each query
			if(freq_r[r] > subWalk[r].size()){
				size_t remain = freq_r[r] - subWalk[r].size();
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
				for(int p = 0; p < remain; ++p){
					subWalk[r].push_back(point2D(IdxJ[p],IdxK[p]));
				}
				free(IdxJ);
				free(IdxK);
			}
			// use the pool of indexes to compute the sampled value
			for(size_t m = 0; m < freq_r[r]; ++m){
				size_t rp = idxRp[offset++];
				size_t idxJ = (subWalk[r])[m].x;
				size_t idxK = (subWalk[r])[m].y;
				double valueSampled = sgn_foo(MatA.GetElement(r,i));
				valueSampled *= sgn_foo(MatB.GetElement(idxJ,r));
				valueSampled *= sgn_foo(MatC.GetElement(idxK,r));
				valueSampled *= sgn_foo(MatA.GetElement(rp,i));
				valueSampled *= MatB.GetElement(idxJ,rp);
				valueSampled *= MatC.GetElement(idxK,rp);
				IrJc[point3D(i,idxJ,idxK)] += valueSampled;
			}
		}
		finish = clock();
		SamplingTime[i] += (double)(finish-start);
		// pre-sorting the scores
		std::vector<pidx3d> sortVec;
		std::vector<pidx3d> tempSortedVec;
		for (auto mapItr = IrJc.begin(); mapItr != IrJc.end(); mapItr++){
			tempSortedVec.push_back(std::make_pair(mapItr->first, mapItr->second));
		}
		start = clock();
		sort(tempSortedVec.begin(),tempSortedVec.end(),compgt<pidx3d>);
		// compute the actual value for top-t' indexes
		for(size_t t = 0; t < tempSortedVec.size() && t < budget; ++t){
			double true_value = vectors_mul(tempSortedVec[t].first, MatA, MatB, MatC);
			sortVec.push_back(std::make_pair(tempSortedVec[t].first, true_value));
		}
		sort(sortVec.begin(),sortVec.end(),compgt<pidx3d>);
		finish = clock();
		SamplingTime[i] += (double)(finish-start);
		SamplingTime[i] /= CLOCKS_PER_SEC;
		for(size_t s = 0; s < sortVec.size() && s < knn; ++s){
			knnValue[i*knn + s] = sortVec[s].second;
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
