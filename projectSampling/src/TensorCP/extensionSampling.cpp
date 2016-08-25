/*
	Extension Sampling with three factor matrices
	usage:
	[value, time, indexes] =  extensionSampling(A, B, C, budget, samples, top_t);

	* Variables input:
		A:	size: (L1, R)
		B:  size: (L2, R)
		C:  size: (L3, R)
		samples: numbers of samples
		top_t : find the top_t value in tensor

		* Variables output:
			value: size: (top_t, 1)
						 the top_t value 
			time: time consuming during the sampling
			indexes: size (top_t, 3)
							 the indexes of the corresponding value	
		Author : Zhi Lu
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
	uint rankSize = mxGetN(prhs[0]);
	uint rankSizeExt = rankSize * rankSize;
	// original matrices
	double *A = mxGetPr(prhs[0]);
	double *B = mxGetPr(prhs[1]);
	double *C = mxGetPr(prhs[2]);
	Matrix MatA(mxGetM(prhs[0]), rankSize, A);
	Matrix MatB(mxGetM(prhs[1]), rankSize, B);
	Matrix MatC(mxGetM(prhs[2]), rankSize, C);
	// the budget
	const uint budget = (uint)mxGetPr(prhs[3])[0];
	// number of samples
	const uint NumSample = (uint)mxGetPr(prhs[4])[0];
	// find the top-t largest value
	const uint top_t = (uint)mxGetPr(prhs[5])[0];
	// result of sampling
	plhs[0] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
	double *plhs_result = mxGetPr(plhs[0]);
	// time duration sampling
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	double *tsec = mxGetPr(plhs[1]);
	// indexes of values
	plhs[2] = mxCreateNumericMatrix(top_t, 3, mxUINT64_CLASS, mxREAL);
	uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[2]);
	mexPrintf("Starting Extension Sampling:");
	mexPrintf("- Top-%d ",top_t);
	mexPrintf("- Samples:1e%d ",(int)log10(NumSample));
	mexPrintf("- Budget:1e%d ",(int)log10(budget));
	mexPrintf("......");mexEvalString("drawnow");
	// compute the extension for matrices
	double *Aex = (double*)malloc(mxGetM(prhs[0])*rankSizeExt*sizeof(double));
	double *Bex = (double*)malloc(mxGetM(prhs[1])*rankSizeExt*sizeof(double));
	double *Cex = (double*)malloc(mxGetM(prhs[2])*rankSizeExt*sizeof(double));
	memset(Aex, 0, mxGetM(prhs[0])*rankSizeExt*sizeof(double));
	memset(Bex, 0, mxGetM(prhs[1])*rankSizeExt*sizeof(double));
	memset(Cex, 0, mxGetM(prhs[2])*rankSizeExt*sizeof(double));
	// compute the extension matrices
	start = clock();
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
	// extension matrices
	Matrix MatAex(mxGetM(prhs[0]), rankSizeExt, Aex);
	Matrix MatBex(mxGetM(prhs[1]), rankSizeExt, Bex);
	Matrix MatCex(mxGetM(prhs[2]), rankSizeExt, Cex);
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec = duration;
	//-------------------------------------
	// Compute weight
	//-------------------------------------
	double SumofW = 0;
	double *weight = (double*)malloc(rankSizeExt*sizeof(double));
	memset(weight, 0, rankSizeExt*sizeof(double));
	start = clock();
	for (uint r = 0; r < rankSizeExt; ++r){
		weight[r] = MatAex.SumofCol[r];
		weight[r] *= MatBex.SumofCol[r];
		weight[r] *= MatCex.SumofCol[r];
		SumofW += weight[r]; 
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	//-------------------------
	// Do Sampling
	//-------------------------
	start = clock();
	uint *freq_r = (uint *)malloc(rankSizeExt*sizeof(uint));
	memset( freq_r, 0, rankSizeExt*sizeof(uint));
	for (uint r = 0; r < rankSizeExt; ++r){
		double u = (double)rand()/(double)RAND_MAX;
		double c = (double)NumSample*weight[r]/SumofW;
		if(u < (c - floor(c)))
			freq_r[r] = (uint)ceil(c);
		else
			freq_r[r] = (uint)floor(c);
	}
	uint *IdxI = (uint*)malloc((NumSample + rankSizeExt)*sizeof(uint));
	memset(IdxI, 0, (NumSample + rankSizeExt)*sizeof(uint));
	uint *IdxJ = (uint*)malloc((NumSample + rankSizeExt)*sizeof(uint));
	memset(IdxJ, 0, (NumSample + rankSizeExt)*sizeof(uint));
	uint *IdxK = (uint*)malloc((NumSample + rankSizeExt)*sizeof(uint));
	memset(IdxK, 0, (NumSample + rankSizeExt)*sizeof(uint));
	// sample indexes
	for (uint r = 0,offset = 0; r < rankSizeExt; ++r){
		// sample i
		vose_alias( freq_r[r], (IdxI + offset), \
					MatAex.row, \
					(MatAex.element + r*MatAex.row), \
					MatAex.SumofCol[r]);
		// sample j
		vose_alias( freq_r[r], (IdxJ + offset), \
					MatBex.row, \
					(MatBex.element + r*MatBex.row), \
					MatBex.SumofCol[r]);	
		// sample k
		vose_alias( freq_r[r], (IdxK + offset), \
					MatCex.row, \
					(MatCex.element + r*MatCex.row), \
					MatCex.SumofCol[r]);						
		offset += freq_r[r];
	}
	// compute update value and saved in map<pair, value>
	// use map IrJc to save the sampled values
	std::map<point3D, double> IrJc;
	for(uint r = 0,offset = 0; r < rankSizeExt; ++r){
		for(uint s = 0; s < freq_r[r]; ++s,++offset){
			uint idxi = IdxI[offset];
			uint idxj = IdxJ[offset];
			uint idxk = IdxK[offset];
			uint idxr = r / rankSize;
			double valueSampled = sgn_foo(MatA.GetElement(idxi,idxr)) \
								* sgn_foo(MatB.GetElement(idxj,idxr)) \
								* sgn_foo(MatC.GetElement(idxk,idxr));
			IrJc[point3D(idxi, idxj, idxk)] += valueSampled;
		}
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	//-----------------------------------
	//sort the values have been sampled
	//-----------------------------------
	// for pre sort
	std::vector<pidx3d> tempSortedVec;
	// sort by actual value
	std::vector<pidx3d> sortVec;
	// push the value into a vector for sorting
	for (auto mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr){
		tempSortedVec.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	start = clock();
	sort(tempSortedVec.begin(), tempSortedVec.end(), compgt<pidx3d>);
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;

	start = clock();
	// compute the top-t' (budget) actual value
	for(uint m = 0; m < tempSortedVec.size() && m < budget; ++m){
		double true_value = MatrixRowMul(tempSortedVec[m].first, MatA, MatB, MatC);
		sortVec.push_back(std::make_pair(tempSortedVec[m].first, true_value));
	}
	// sort the vector according to the actual value
	sort(sortVec.begin(), sortVec.end(), compgt<pidx3d>);
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	//--------------------------------
	// Converting to Matlab
	//--------------------------------
	// value
	for(uint m = 0; m < sortVec.size() && m < top_t; ++m){
		//value
		plhs_result[m] = sortVec[m].second;
		//i
		plhs_pr[m] = (sortVec[m].first.x + 1);
		//j
		plhs_pr[m + top_t] = (sortVec[m].first.y + 1);
		//k
		plhs_pr[m + top_t + top_t] = (sortVec[m].first.z + 1);
	}
	mexPrintf("Done!\n");
	//---------------
	// free
	//---------------
	free(weight);
	free(IdxI);
	free(IdxJ);
	free(IdxK);
	free(freq_r);
	free(Aex);
	free(Bex);
	free(Cex);
}
