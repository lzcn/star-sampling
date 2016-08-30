/*
	Diamond Sampling for queries
	[value, time] = querySampling(A, B, C, budget, samples, knn)
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
	srand(unsigned(time(NULL)));
	//--------------------
	// Initialization
	//--------------------
	uint rankSize = mxGetN(prhs[0]);
	uint rankSizeExt = rankSize * rankSize;
	// normal matrix
	double *A = mxGetPr(prhs[0]);
	double *B = mxGetPr(prhs[1]);
	double *C = mxGetPr(prhs[2]);
	Matrix MatA(mxGetM(prhs[0]), rankSize, A,MATRIX_NONE_SUM);
	Matrix MatB(mxGetM(prhs[1]), rankSize, B,MATRIX_NONE_SUM);
	Matrix MatC(mxGetM(prhs[2]), rankSize, C,MATRIX_NONE_SUM);
	// the budget
	const size_t budget = (size_t)mxGetPr(prhs[3])[0];
	// number of samples
	const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
	// find the top-t largest value
	const uint knn = (uint)mxGetPr(prhs[5])[0];
	// number of queries
	const uint NumQueries = mxGetM(prhs[0]);
	// result values for each query
	plhs[0] = mxCreateDoubleMatrix(knn, NumQueries, mxREAL);
	double *knnValue = mxGetPr(plhs[0]);
	// sampling time for each query
	plhs[1] = mxCreateDoubleMatrix(NumQueries, 1, mxREAL);	
	double *SamplingTime = mxGetPr(plhs[1]);
	memset(SamplingTime, 0, NumQueries*sizeof(double));
	mexPrintf("Central Sampling for Queries");
	mexPrintf("- Top-%d ",knn);
	mexPrintf("- Samples:1e%d ",(int)log10(NumSample));
	mexPrintf("- Budget:1e%d ",(int)log10(budget));
	mexPrintf("- Number of Queries:%d ",(NumQueries));
	mexPrintf("......\n");mexEvalString("drawnow");
	//-------------------------
	// extension for matrices
	//-------------------------
	start = clock();
	double *Aex = (double*)malloc(mxGetM(prhs[0])*rankSizeExt*sizeof(double));
	double *Bex = (double*)malloc(mxGetM(prhs[1])*rankSizeExt*sizeof(double));
	double *Cex = (double*)malloc(mxGetM(prhs[2])*rankSizeExt*sizeof(double));
	memset(Aex, 0, mxGetM(prhs[0])*rankSizeExt*sizeof(double));
	memset(Bex, 0, mxGetM(prhs[1])*rankSizeExt*sizeof(double));
	memset(Cex, 0, mxGetM(prhs[2])*rankSizeExt*sizeof(double));
	for (uint m = 0; m < rankSize; ++m){
		for (uint n = 0; n < rankSize; ++n){
			// extension for matrix A
			for(uint i = 0; i < mxGetM(prhs[0]); ++i){
				Aex[(m*rankSize + n) * MatA.row + i] = A[m * MatA.row + i] * A[n * MatA.row + i];
			}
			// extension for matrix B
			for(uint j = 0; j < mxGetM(prhs[1]); ++j){
				Bex[(m*rankSize + n) * MatB.row + j] = B[m * MatB.row + j] * B[n * MatB.row + j];
			}
			// extension for matrix C
			for(uint k = 0; k < mxGetM(prhs[2]); ++k){
				Cex[(m*rankSize + n) * MatC.row + k] = C[m * MatC.row + k] * C[n * MatC.row + k];
			}
		}
	}
	finish = clock();
	for (size_t i = 0; i < NumQueries; i++) {
		SamplingTime[i] = (double)(finish - start)/NumQueries;
	}
	// extension matrices
	Matrix MatAex(mxGetM(prhs[0]), rankSizeExt, Aex,MATRIX_COL_SUM);
	Matrix MatBex(mxGetM(prhs[1]), rankSizeExt, Bex,MATRIX_COL_SUM);
	Matrix MatCex(mxGetM(prhs[2]), rankSizeExt, Cex,MATRIX_COL_SUM);
	//-------------------------
	// Do Sampling
	//-------------------------
	double *weight = (double*)malloc(rankSizeExt*sizeof(double));
	memset(weight, 0, rankSizeExt*sizeof(double));
	size_t *freq_r = (size_t *)malloc(NumQueries*rankSizeExt*sizeof(size_t));
	memset( freq_r, 0, NumQueries*rankSizeExt*sizeof(size_t));
	//-----------------------
	// Sampling the Indexes
	//-----------------------
	progressbar(0);
	std::vector<std::vector<point2D>> subWalk(rankSizeExt);
	for(uint i = 0; i < NumQueries; ++i){
		clearprogressbar();
		progressbar((double)i/NumQueries);
		double SumofW = 0.0;
		start = clock();
		for (uint r = 0; r < rankSizeExt; ++r){
			weight[r] = abs(MatAex.GetElement(i,r));
			//weight[r] *= MatBex.SumofCol[r];
			//weight[r] *= MatCex.SumofCol[r];
			SumofW += weight[r]; 
		}
		finish = clock();
		SamplingTime[i] += (double)(finish - start);
		if(SumofW == 0){
			continue;
		}
		// sample freq_r[r]
		start = clock();
		for (uint r = 0; r < rankSizeExt; ++r){
			double u = (double)rand()/(double)RAND_MAX;
			double c = (double)NumSample*weight[r]/SumofW;
			if(u < (c - floor(c)))
				freq_r[r] = (size_t)ceil(c);
			else
				freq_r[r] = (size_t)floor(c);
		}
		std::map<point3D, double> IrJc;
		for(uint r = 0; r < rankSizeExt; ++r){
			if(freq_r[r] > subWalk[r].size()){
				size_t remain = freq_r[r] - subWalk[r].size();
				uint *IdxJ = (uint*)malloc(remain*sizeof(uint));
				uint *IdxK = (uint*)malloc(remain*sizeof(uint));
				memset(IdxJ, 0, remain*sizeof(uint));
				memset(IdxK, 0, remain*sizeof(uint));
				// sample indexes
				vose_alias( remain, IdxJ, \
							MatBex.row, \
							(MatBex.element + r*MatBex.row), \
							MatBex.SumofCol[r]);
				vose_alias( remain, IdxK, \
							MatCex.row, \
							(MatCex.element + r*MatCex.row), \
							MatCex.SumofCol[r]);
			    for(uint p = 0; p < remain; ++p){
					subWalk[r].push_back(point2D(IdxJ[p],IdxK[p]));
				}
				free(IdxJ);
				free(IdxK);
			}
			for(size_t m = 0; m < freq_r[r]; ++m){
				uint idxJ = (subWalk[r])[m].x;
				uint idxK = (subWalk[r])[m].y;
				double valueSampled = sgn_foo(MatAex.GetElement(i,r));
				valueSampled *= sgn_foo(MatBex.GetElement(idxJ,r));
				valueSampled *= sgn_foo(MatCex.GetElement(idxK,r));
				IrJc[point3D(i,idxJ,idxK)] += valueSampled;
			}
		}
		finish = clock();
		SamplingTime[i] += (double)(finish - start);
		//-----------------------------------
		//sort the values have been sampled
		//-----------------------------------
		std::vector<pidx3d> tempSortedVec;
		for (auto mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr) {
			tempSortedVec.push_back(std::make_pair(mapItr->first, mapItr->second));
		}
		start = clock();
		sort(tempSortedVec.begin(), tempSortedVec.end(), compgt<pidx3d>);
		
		std::vector<pidx3d> sortVec;
		for(size_t t = 0; t < tempSortedVec.size() && t < budget; ++t){
			double true_value = MatrixRowMul(tempSortedVec[t].first, MatA, MatB, MatC);
			sortVec.push_back(std::make_pair(tempSortedVec[t].first,true_value));
		}
		sort(sortVec.begin(),sortVec.end(),compgt<pidx3d>);
		finish = clock();
		SamplingTime[i] += (double)(finish-start);
		SamplingTime[i] /= (double)CLOCKS_PER_SEC;
		for(size_t s = 0; s < sortVec.size() && s < knn; ++s){
			knnValue[i*knn + s] = sortVec[s].second;
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
	free(Aex);
	free(Bex);
	free(Cex);
}
