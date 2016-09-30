#include <vector>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <ctime>

#include "mex.h"
#include "utilmex.h"
#include "matrix.h"

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	clock_t start,finish;
	srand(unsigned(time(NULL)));
	//--------------------
	// Initialization
	//--------------------
	uint Arow = (uint)mxGetM(prhs[0]);
	uint Brow = (uint)mxGetM(prhs[1]);
	uint Crow = (uint)mxGetM(prhs[2]);
	uint rankSize = (uint)mxGetN(prhs[0]);
	uint rankSizeExt = rankSize * rankSize;
	// normal matrix
	double *A = mxGetPr(prhs[0]);
	double *B = mxGetPr(prhs[1]);
	double *C = mxGetPr(prhs[2]);
	// the budget
	const size_t budget = (size_t)mxGetPr(prhs[3])[0];
	// number of samples
	const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
	// find the top-t largest value
	const uint knn = (uint)mxGetPr(prhs[5])[0];
	// number of queries
	const uint NumQueries = (uint)mxGetM(prhs[0]);
	// result values for each query
	plhs[0] = mxCreateDoubleMatrix(knn, NumQueries, mxREAL);
	double *knnValue = mxGetPr(plhs[0]);
	// sampling time for each query
	plhs[1] = mxCreateDoubleMatrix(NumQueries, 4, mxREAL);	
	double *SamplingTime = mxGetPr(plhs[1]);
	memset(SamplingTime, 0, NumQueries*sizeof(double));
	mexPrintf("Central Sampling for Queries:");
	mexPrintf("Top:%d,Samples:1e%d,Budget:1e%d,Number of Queries:%d\n",knn,(int)log10(NumSample),(int)log10(budget),NumQueries);
	mexEvalString("drawnow");
	//-------------------------
	// extension for matrices
	//-------------------------
	start = clock();
	Matrix AT(rankSize,Arow);
	Matrix BT(rankSize,Brow);
	Matrix CT(rankSize,Crow);
	AT.transpose(A);
	BT.transpose(B);
	CT.transpose(C);
	double *Aex = (double*)malloc(Arow*rankSizeExt*sizeof(double));
	double *Bex = (double*)malloc(Brow*rankSizeExt*sizeof(double));
	double *Cex = (double*)malloc(Crow*rankSizeExt*sizeof(double));
	memset(Aex, 0, Arow*rankSizeExt*sizeof(double));
	memset(Bex, 0, Brow*rankSizeExt*sizeof(double));
	memset(Cex, 0, Crow*rankSizeExt*sizeof(double));
	for (uint m = 0; m < rankSize; ++m){
		for (uint n = 0; n < rankSize; ++n){
			size_t r = m * rankSize + n;
			double sum = 0;
			for(uint i = 0; i < Arow; ++i){
				sum += abs(A[m * Arow + i] * A[n * Arow + i]);
				Aex[r * Arow + i] = sum;
			}
			sum = 0;
			for(uint j = 0; j < Brow; ++j){
				sum += abs(B[m * Brow + j] * B[n * Brow + j]);
				Bex[r * Brow + j] = sum;
			}
			sum = 0;
			for(uint k = 0; k < Crow; ++k){
				sum += abs(C[m * Crow + k] * C[n * Crow + k]);
				Cex[r * Crow + k] = sum;
			}
		}
	}
	Matrix MatAex(Arow, rankSizeExt, Aex, MATRIX_NONE_SUM);
	Matrix MatBex(Brow, rankSizeExt, Bex, MATRIX_NONE_SUM);
	Matrix MatCex(Crow, rankSizeExt, Cex, MATRIX_NONE_SUM);
	finish = clock();
	for (uint i = 0; i < NumQueries; i++) {
		SamplingTime[i] = (double)(finish - start)/(CLOCKS_PER_SEC*NumQueries);
	}
	// extension matrices
	//-------------------------
	// Do Sampling
	//-------------------------
	double *weight = (double*)malloc(rankSizeExt*sizeof(double));
	memset(weight, 0, rankSizeExt*sizeof(double));
	size_t *freq_r = (size_t *)malloc(NumQueries*rankSizeExt*sizeof(size_t));
	memset( freq_r, 0, NumQueries*rankSizeExt*sizeof(size_t));
	uint *IdxJ = (uint*)malloc(NumSample*sizeof(uint));
	uint *IdxK = (uint*)malloc(NumSample*sizeof(uint));
	memset(IdxJ, 0, NumSample*sizeof(uint));
	memset(IdxK, 0, NumSample*sizeof(uint));
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
			weight[r] = MatAex(Arow - 1, r);
			weight[r] *= MatBex(Brow - 1, r);
			weight[r] *= MatCex(Crow - 1, r);
			SumofW += weight[r]; 
		}
		finish = clock();
		SamplingTime[i] += (double)(finish - start)/CLOCKS_PER_SEC;
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
		finish = clock();
		SamplingTime[NumQueries + i] = (double)(finish - start)/CLOCKS_PER_SEC;
		//std::map<point3D, double> IrJc;
		TPoint3DMap IrJc;
		for(uint r = 0; r < rankSizeExt; ++r){
			start = clock();
			if(freq_r[r] > subWalk[r].size()){
				size_t remain = freq_r[r] - subWalk[r].size();
				// sample indexes
				binary_search( remain, IdxJ, Brow,(MatBex.element + r*Brow));
				binary_search( remain, IdxK, Crow,(MatCex.element + r*Crow));
			    for(uint p = 0; p < remain; ++p){
					subWalk[r].push_back(point2D(IdxJ[p],IdxK[p]));
				}
			}
			finish = clock();
			SamplingTime[NumQueries + i] += (double)(finish - start)/CLOCKS_PER_SEC;
			start = clock();
			for(size_t m = 0; m < freq_r[r]; ++m){
				uint idxJ = (subWalk[r])[m].x;
				uint idxK = (subWalk[r])[m].y;
				IrJc[point3D(i,idxJ,idxK)] += sgn(MatAex(i,r))*sgn(MatBex(idxJ,r))*sgn(MatCex(idxK,r));
			}
			finish = clock();
			SamplingTime[2*NumQueries + i] = (double)(finish - start)/CLOCKS_PER_SEC;
		}
		//-----------------------------------
		//sort the values have been sampled
		//-----------------------------------
		start = clock();
		std::vector<pidx3d> tempSortedVec;
		std::vector<pidx3d> sortVec;
		for (auto mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr) {
			tempSortedVec.push_back(std::make_pair(mapItr->first, mapItr->second));
		}
		sort(tempSortedVec.begin(), tempSortedVec.end(), compgt<pidx3d>);
		for(size_t t = 0; t < tempSortedVec.size() && t < budget; ++t){
			double true_value = MatrixColMul(tempSortedVec[t].first, AT, BT, CT);
			sortVec.push_back(std::make_pair(tempSortedVec[t].first,true_value));
		}
		sort(sortVec.begin(),sortVec.end(),compgt<pidx3d>);
		finish = clock();
		SamplingTime[3*NumSample + i] = (double)(finish-start)/CLOCKS_PER_SEC;
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
	free(IdxJ);
	free(IdxK);
}
