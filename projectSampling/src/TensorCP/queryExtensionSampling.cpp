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

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	clock_t start,finish;
	srand(unsigned(time(NULL)));
	//--------------------
	// Initialization
	//--------------------
	start = clock();
	size_t rankSize = mxGetN(prhs[0]);
	size_t rankSizeExt = rankSize * rankSize;
	// normal matrix
	double *A = mxGetPr(prhs[0]);
	double *B = mxGetPr(prhs[1]);
	double *C = mxGetPr(prhs[2]);
	Matrix MatA(mxGetM(prhs[0]), rankSize, A);
	Matrix MatB(mxGetM(prhs[1]), rankSize, B);
	Matrix MatC(mxGetM(prhs[2]), rankSize, C);
	// extension for matrices
	double *Aex = (double*)malloc(mxGetM(prhs[0])*rankSizeExt*sizeof(double));
	double *Bex = (double*)malloc(mxGetM(prhs[1])*rankSizeExt*sizeof(double));
	double *Cex = (double*)malloc(mxGetM(prhs[2])*rankSizeExt*sizeof(double));
	memset(Aex, 0, mxGetM(prhs[0])*rankSizeExt*sizeof(double));
	memset(Bex, 0, mxGetM(prhs[1])*rankSizeExt*sizeof(double));
	memset(Cex, 0, mxGetM(prhs[2])*rankSizeExt*sizeof(double));

	for (size_t m = 0; m < rankSize; ++m){
		for (size_t n = 0; n < rankSize; ++n){
			// extension for matrix A
			for(size_t i = 0; i < mxGetM(prhs[0]); ++i){
				Aex[(m*rankSize + n) * MatA.row + i] = A[m * MatA.row + i] * A[n * MatA.row + i];
			}
			// extension for matrix B
			for(size_t j = 0; j < mxGetM(prhs[1]); ++j){
				Bex[(m*rankSize + n) * MatB.row + j] = B[m * MatB.row + j] * B[n * MatB.row + j];
			}
			// extension for matrix C
			for(size_t k = 0; k < mxGetM(prhs[2]); ++k){
				Cex[(m*rankSize + n) * MatC.row + k] = C[m * MatC.row + k] * C[n * MatC.row + k];
			}
		}
	}
	// extension matrices
	Matrix MatAex(mxGetM(prhs[0]), rankSizeExt, Aex);
	Matrix MatBex(mxGetM(prhs[1]), rankSizeExt, Bex);
	Matrix MatCex(mxGetM(prhs[2]), rankSizeExt, Cex);
	// the budget
	const size_t budget = (size_t)mxGetPr(prhs[3])[0];
	// number of samples
	const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
	// find the top-t largest value
	const size_t knn = (size_t)mxGetPr(prhs[5])[0];
	// number of queries
	const size_t NumQueries = mxGetM(prhs[0]);
	// result values for each query
	plhs[0] = mxCreateDoubleMatrix(knn, NumQueries, mxREAL);
	double *knnValue = mxGetPr(plhs[0]);
	
	// sampling time for each query
	plhs[1] = mxCreateDoubleMatrix(NumQueries, 1, mxREAL);	
	double *SamplingTime = mxGetPr(plhs[1]);
	memset(SamplingTime, 0, NumQueries*sizeof(double));
	finish = clock();
	for (size_t i = 0; i < NumQueries; ++i) {
		SamplingTime[i] = (double)(finish-start)/NumQueries;
	}
	mexPrintf(">> Initialization Complete!\n");


	//-------------------------------------
	// Compute weight
	//-------------------------------------
	double *weight = (double*)malloc(NumQueries*rankSizeExt*sizeof(double));
	memset(weight, 0, NumQueries*rankSizeExt*sizeof(double));
	double *SumofW = (double*)malloc(NumQueries*sizeof(double));
	memset(SumofW, 0, NumQueries*sizeof(double));
	for (size_t i = 0; i < NumQueries; ++i) {
		start = clock();
		SumofW[i] = 0;
		for (size_t r = 0; r < rankSizeExt; ++r){
			double tempW;
			tempW = abs(MatAex.GetElement(i,r));
			tempW *= MatBex.SumofCol[r];
			tempW *= MatCex.SumofCol[r];
			weight[i*rankSizeExt + r] = tempW;
			SumofW[i] += tempW; 
		}
		finish = clock();
		SamplingTime[i] += (double)(finish-start);
	}
	mexPrintf(">> Computing weight complete!\n");
	//-------------------------
	// Do Sampling
	//-------------------------
	mexPrintf(">> Start computing c_r!\n");
	size_t *freq_r = (size_t *)malloc(NumQueries*rankSizeExt*sizeof(size_t));
	memset( freq_r, 0, NumQueries*rankSizeExt*sizeof(size_t));
	for (size_t i = 0; i < NumQueries; ++i){
		start = clock();
		for (size_t r = 0; r < rankSizeExt; ++r){
			double u = (double)rand()/(double)RAND_MAX;
			double c = (double)NumSample*weight[i*rankSizeExt + r]/SumofW[i];
			if(u < (c - floor(c)))
				freq_r[i*rankSizeExt + r] = (size_t)ceil(c);
			else
				freq_r[i*rankSizeExt + r] = (size_t)floor(c);
		}
		finish = clock();
		SamplingTime[i] += (double)(finish - start);
	}
	//-----------------------
	// Sampling the Indexes
	//-----------------------
	mexPrintf(">> Start Sampling Indexes!\n");
	std::vector<std::vector<point2D>> subWalk(rankSizeExt);
	for(size_t i = 0; i < NumQueries; ++i){
		start = clock();
		std::map<point3D, double> IrJc;
		for(size_t r = 0; r < rankSizeExt; ++r){
			if(freq_r[i*rankSizeExt + r] > subWalk[r].size()){
				size_t remain = freq_r[i*rankSizeExt + r] - subWalk[r].size();
				size_t *IdxJ = (size_t*)malloc(remain*sizeof(size_t));
				size_t *IdxK = (size_t*)malloc(remain*sizeof(size_t));
				memset(IdxJ, 0, remain*sizeof(size_t));
				memset(IdxK, 0, remain*sizeof(size_t));
				// sample indexes
				vose_alias( remain, IdxJ, \
							MatBex.row, \
							(MatBex.element + r*MatBex.row), \
							MatBex.SumofCol[r]);
				vose_alias( remain, IdxK, \
							MatCex.row, \
							(MatCex.element + r*MatCex.row), \
							MatCex.SumofCol[r]);
			    for(size_t p = 0; p < remain; ++p){
					subWalk[r].push_back(point2D(IdxJ[p],IdxK[p]));
				}
				free(IdxJ);
				free(IdxK);
			}
			for(size_t m = 0; m < freq_r[i*rankSizeExt + r]; ++m){
				size_t idxJ = (subWalk[r])[m].x;
				size_t idxK = (subWalk[r])[m].y;
				double valueSampled = 1.0;
				valueSampled *= sgn_foo(MatAex.GetElement(i,r));
				valueSampled *= sgn_foo(MatBex.GetElement(idxJ,r));
				valueSampled *= sgn_foo(MatCex.GetElement(idxK,r));
				IrJc[point3D(i,idxJ,idxK)] += valueSampled;
			}
		}
		//-----------------------------------
		//sort the values have been sampled
		//-----------------------------------
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
	mexPrintf(">> Done!\n");
	//---------------
	// free
	//---------------
	free(weight);
	free(SumofW);
	free(freq_r);
	free(Aex);
	free(Bex);
	free(Cex);
}
