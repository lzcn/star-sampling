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
	uint Arow = mxGetM(prhs[0]);
	uint Brow = mxGetM(prhs[1]);
	uint Crow = mxGetM(prhs[2]);
	start = clock();
	Matrix MatA(mxGetM(prhs[0]), mxGetN(prhs[0]), mxGetPr(prhs[0]), MATRIX_NONE_SUM);
	Matrix MatB(mxGetM(prhs[1]), mxGetN(prhs[1]), mxGetPr(prhs[1]), MATRIX_NONE_SUM);
	Matrix MatC(mxGetM(prhs[2]), mxGetN(prhs[2]), mxGetPr(prhs[2]), MATRIX_NONE_SUM);
	Matrix AT(mxGetN(prhs[0]),mxGetM(prhs[0]));
	Matrix BT(mxGetN(prhs[1]),mxGetM(prhs[1]));
	Matrix CT(mxGetN(prhs[2]),mxGetM(prhs[2]));
	for(uint r = 0 ; r < rankSize; ++r){
		for (uint i = 0; i < MatA.row; ++i) {
			AT(r,i) = MatA(i,r);
		}
		for (uint j = 0; j < MatB.row; ++j) {
			BT(r,j) = MatB(j,r);
		}
		for (uint k = 0; k < MatC.row; ++k) {
			CT(r,k) = MatC(k,r);
		}
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	// the budget
	const size_t budget = (size_t)mxGetPr(prhs[3])[0];
	// number of samples
	const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
	// find the top-t largest value
	const uint top_t = (uint)mxGetPr(prhs[5])[0];
	// result of sampling
	plhs[0] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
	double *plhs_result = mxGetPr(plhs[0]);
	// time duration sampling
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	double *tsec = mxGetPr(plhs[1]);
	*tsec = duration;
	// indexes of values
	plhs[2] = mxCreateNumericMatrix(top_t, 3, mxUINT64_CLASS, mxREAL);
	uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[2]);
	mexPrintf("Starting Core^2 Diamond Sampling:");
	mexPrintf("- Top:%d ",top_t);
	mexPrintf("- Samples:1e%d ",(int)log10(NumSample));
	mexPrintf("- Budget:1e%d ",(int)log10(budget));
	mexPrintf("......\n");mexEvalString("drawnow");
	// compute the extension for matrices
	start = clock();
	double *Aex = (double*)malloc(mxGetM(prhs[0])*rankSizeExt*sizeof(double));
	double *Bex = (double*)malloc(mxGetM(prhs[1])*rankSizeExt*sizeof(double));
	double *Cex = (double*)malloc(mxGetM(prhs[2])*rankSizeExt*sizeof(double));
	memset(Aex, 0, mxGetM(prhs[0])*rankSizeExt*sizeof(double));
	memset(Bex, 0, mxGetM(prhs[1])*rankSizeExt*sizeof(double));
	memset(Cex, 0, mxGetM(prhs[2])*rankSizeExt*sizeof(double));
	// compute the extension matrices
	// extension for matrix A
	for(uint i = 0; i < Arow; ++i){
		double sum = 0;
		for (uint m = 0; m < rankSize; ++m){
			for (uint n = 0; n < rankSize; ++n){
				sum += abs(A[m * Arow + i] * A[n * Arow + i]);
				Aex[m * rankSize + n + i*rankSizeExt] = sum;
			}
		}
	}
	for (uint m = 0; m < rankSize; ++m){
		for (uint n = 0; n < rankSize; ++n){
			size_t r = m * rankSize + n;
			double sum = 0.0;
			// extension for matrix B
			for(uint j = 0; j < Brow; ++j){
				sum += abs(B[m * Brow + j] * B[n * Brow + j]);
				Bex[r * Brow + j] = sum;
			}
			// extension for matrix C
			sum = 0;
			for(uint k = 0; k < Crow; ++k){
				sum += abs(C[m * Crow + k] * C[n * Crow + k]);
				Cex[r * Crow + k] = sum;
			}
		}
	}
	// extension matrices
	Matrix MatAex(rankSizeExt, mxGetM(prhs[0]), Aex, MATRIX_NONE_SUM);
	Matrix MatBex(mxGetM(prhs[1]), rankSizeExt, Bex, MATRIX_NONE_SUM);
	Matrix MatCex(mxGetM(prhs[2]), rankSizeExt, Cex, MATRIX_NONE_SUM);
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	//-------------------------------------
	// Compute weight
	//-------------------------------------
	double SumofW = 0;
	double *weight = (double*)malloc(Arow*rankSizeExt*sizeof(double));
	memset(weight, 0, Arow*rankSizeExt*sizeof(double));
	start = clock();
	for (uint m = 0; m < rankSize; ++m){
		for (uint n = 0; n < rankSize; ++n){
				for(uint i = 0; i < Arow; ++i){
					size_t r = m * rankSize + n;
					double tempW = abs(A[m * Arow + i] * A[n * Arow + i]);
					tempW *= MatAex(rankSizeExt - 1, i);
					tempW *= MatBex(Brow - 1, r);
					tempW *= MatCex(Crow - 1, r);
					weight[r*Arow + i] = tempW; 
					SumofW += tempW;
				}
		}
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	mexPrintf("|-%f during the initialization phase.\n",*tsec);mexEvalString("drawnow");
	//-------------------------
	// Do Sampling
	//-------------------------
	start = clock();
	uint *IdxI = (uint*)malloc(NumSample*sizeof(uint));
	memset(IdxI, 0, NumSample*sizeof(uint));
	uint *IdxJ = (uint*)malloc(NumSample*sizeof(uint));
	memset(IdxJ, 0, NumSample*sizeof(uint));
	uint *IdxK = (uint*)malloc(NumSample*sizeof(uint));
	memset(IdxK, 0, NumSample*sizeof(uint));
	uint *IdxR = (uint*)malloc(NumSample*sizeof(uint));
	memset(IdxR, 0, NumSample*sizeof(uint));	
	size_t *freq_r = (size_t *)malloc(rankSizeExt*sizeof(size_t));
	memset( freq_r, 0, rankSizeExt*sizeof(size_t));
	sort_sample(NumSample, \
				 IdxI, IdxR, \
				 freq_r, \
				 MatAex.row, MatAex.col, \
				 weight, SumofW);
	// sample indexes
	size_t offset = 0;
	for (uint r = 0; r < rankSizeExt; ++r){
		// sample j
		binary_search( freq_r[r], (IdxJ + offset), Brow, (MatBex.element + r*Brow));
		// sample k
		binary_search( freq_r[r], (IdxK + offset), Crow, (MatCex.element + r*Crow));
		offset += freq_r[r];
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	mexPrintf("|-%f during the sampling phase.\n",duration);mexEvalString("drawnow");
	// compute update value and saved in map<pair, value>
	// use map IrJc to save the sampled values
	start = clock();
	// std::map<point3D, double> IrJc;
	TPoint3DMap IrJc;
	for (size_t s = 0; s < NumSample ; ++s){
		uint i = IdxI[s];
		uint j = IdxJ[s];
		uint k = IdxK[s];
		uint r = IdxR[s];
		double u = MatAex(rankSizeExt - 1,i)*((double)rand()/(double)RAND_MAX);
		uint rp = binary_search_once((MatAex.element + i*rankSizeExt),Arow - 1,u);
		// Update the element in coordinate
		IrJc[point3D(i, j, k)] += sgn(MatAex(r,i))*sgn(MatBex(j,r))*sgn(MatCex(k,r))*sgn(MatAex(rp,i))*MatBex(j,rp)*MatCex(k,rp);
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	mexPrintf("|-%f during the scoring phase.\n",duration);mexEvalString("drawnow");
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
	for(size_t m = 0; m < tempSortedVec.size() && m < budget; ++m){
		double true_value = MatrixColMul(tempSortedVec[m].first, AT, BT, CT);
		sortVec.push_back(std::make_pair(tempSortedVec[m].first, true_value));
	}
	// sort the vector according to the actual value
	sort(sortVec.begin(), sortVec.end(), compgt<pidx3d>);
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	mexPrintf("|-%f during the sorting phase.\n",duration);mexEvalString("drawnow");
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
	free(IdxR);
	free(freq_r);
	free(Aex);
	free(Bex);
	free(Cex);
}
