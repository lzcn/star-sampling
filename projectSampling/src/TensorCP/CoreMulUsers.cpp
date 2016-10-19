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
#include "utilmex.h"
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	clock_t start,finish;
	double duration;
	srand(unsigned(time(NULL)));
	//--------------------
	// Initialization
	//--------------------
	// number of queries
	const uint NumQueries = mxGetM(prhs[0]);
	// rank size
	const uint rankSize = mxGetN(prhs[0]);
	// MatA is a set of queries
	Matrix MatA(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
	Matrix MatB(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
	Matrix MatC(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));
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
	memset(SamplingTime, 0, NumQueries*sizeof(double));
	mexPrintf("Central Sampling for Queries");
	mexPrintf("- Top:%d ",knn);
	mexPrintf("- Samples:1e%d ",(int)log10(NumSample));
	mexPrintf("- Budget:1e%d ",(int)log10(budget));
	mexPrintf("- Number of Queries:%d ",(NumQueries));
	mexPrintf("......\n");mexEvalString("drawnow");
	//-------------------------------------
	// Compute weight
	//-------------------------------------
	// weight for each query
	double *weight = (double*)malloc(rankSize*sizeof(double));
	memset(weight, 0,rankSize*sizeof(double));
	size_t *freq_r = (size_t*)malloc(rankSize*sizeof(size_t));
	memset(freq_r, 0, rankSize*sizeof(size_t));
	//-------------------------
	// Do Sampling
	//-------------------------
	// list for sub walk
	std::vector<std::vector<point2D> > subWalk(rankSize);
	progressbar(0);
	for(uint i = 0; i < NumQueries; ++i){
		clearprogressbar();
		progressbar((double)i/NumQueries);
		start = clock();
		double SumofW = 0.0;
		for (uint r = 0; r < rankSize; ++r){
			weight[r] = abs(MatA.GetElement(i,r));
			weight[r] *= MatB.SumofCol[r];
			weight[r] *= MatC.SumofCol[r];
			SumofW += weight[r];
		}
		finish = clock();
		SamplingTime[i] = (double)(finish - start);
		if(SumofW == 0)
			continue;
		start = clock();
		std::map<point3D, double> IrJc;
		for (uint r = 0; r < rankSize; ++r){
			double u = (double)rand()/(double)RAND_MAX;
			double c = (double)NumSample*weight[r]/SumofW;
			if(u < (c - floor(c)))
				freq_r[r] = (size_t)ceil(c);
			else
				freq_r[r] = (size_t)floor(c);
		}
		for(uint r = 0; r < rankSize; ++r){
			// Check the list length for each query
			if(freq_r[r] > subWalk[r].size()){
				size_t remain = freq_r[r] - subWalk[r].size();
				uint *IdxJ = (uint*)malloc(remain*sizeof(uint));
				uint *IdxK = (uint*)malloc(remain*sizeof(uint));
				memset(IdxJ, 0, remain*sizeof(uint));
				memset(IdxK, 0, remain*sizeof(uint));
				vose_alias(remain, IdxJ, \
					MatB.row, \
					(MatB.element + r*MatB.row), \
					MatB.SumofCol[r]);
				vose_alias(remain, IdxK, \
					MatC.row, \
					(MatC.element + r*MatC.row), \
					MatC.SumofCol[r]);
				for(uint p = 0; p < remain; ++p){
					subWalk[r].push_back(point2D(IdxJ[p],IdxK[p]));
				}
				free(IdxJ);
				free(IdxK);
			}
			for(size_t m = 0; m < freq_r[r]; ++m){
				// repeat c_r times to sample indexes j, k
				uint idxJ = (subWalk[r])[m].x;
				uint idxK = (subWalk[r])[m].y;
				double valueSampled;
				valueSampled = sgn_foo(MatA.GetElement(i,r));
				valueSampled *= sgn_foo(MatB.GetElement(idxJ,r));
				valueSampled *= sgn_foo(MatC.GetElement(idxK,r));
				IrJc[point3D(i,idxJ,idxK)] += valueSampled;
			}
		}
		finish = clock();
		SamplingTime[i] += (double)(finish - start);
		//-----------------------------------
		//sort the values have been sampled
		//-----------------------------------
		std::vector<pidx3d> tempSortedVec;
		std::vector<pidx3d> sortVec;
		for (auto mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr) {
			tempSortedVec.push_back(std::make_pair(mapItr->first, mapItr->second));
		}
		start = clock();
		sort(tempSortedVec.begin(), tempSortedVec.end(), compgt<pidx3d>);
		for(uint t = 0; t < tempSortedVec.size() && t < budget; ++t){
			double true_value = MatrixRowMul(tempSortedVec[t].first, MatA, MatB, MatC);
			sortVec.push_back(std::make_pair(tempSortedVec[t].first,true_value));
		}
		sort(sortVec.begin(),sortVec.end(),compgt<pidx3d>);
		finish = clock();
		SamplingTime[i] += (double)(finish-start);
		SamplingTime[i] /= (double)CLOCKS_PER_SEC;
		for(uint s = 0; s < sortVec.size() && s < knn; ++s){
			knnValue[i*knn + s] = sortVec[s].second;
		}
	}
	clearprogressbar();
	progressbar(1);
	//---------------
	// free
	//---------------
	free(weight);
	free(freq_r);
}