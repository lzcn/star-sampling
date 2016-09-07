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
	uint rankSize = (uint)mxGetN(prhs[0]);
	uint rankSizeExt = rankSize * rankSize * rankSize;
	uint Arow = (uint)mxGetM(prhs[0]);
	uint Brow = (uint)mxGetM(prhs[1]);
	// original matrices
	start = clock();
	Matrix MatA(Arow, rankSize, mxGetPr(prhs[0]), MATRIX_NONE_SUM);
	Matrix MatB(Brow, rankSize, mxGetPr(prhs[1]), MATRIX_NONE_SUM);
	Matrix AT(mxGetN(prhs[0]),mxGetM(prhs[0]));
	Matrix BT(mxGetN(prhs[1]),mxGetM(prhs[1]));
	for(uint r = 0 ; r < rankSize; ++r){
		for (uint i = 0; i < MatA.row; ++i) {
			AT(r,i) = MatA(i,r);
		}
		for (uint j = 0; j < MatB.row; ++j) {
			BT(r,j) = MatB(j,r);
		}
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	// the budget
	const size_t budget = (size_t)mxGetPr(prhs[2])[0];
	// number of samples
	const size_t NumSample = (size_t)mxGetPr(prhs[3])[0];
	// find the top-t largest value
	const uint top_t = (uint)mxGetPr(prhs[4])[0];
	// result of sampling
	plhs[0] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
	double *plhs_result = mxGetPr(plhs[0]);
	// time duration sampling
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	double *tsec = mxGetPr(plhs[1]);
	*tsec = duration;
	// indexes of values
	plhs[2] = mxCreateNumericMatrix(top_t, 2, mxUINT64_CLASS, mxREAL);
	uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[2]);
	mexPrintf("Starting Core^3 Sampling:");
	mexPrintf("- Top:%d ",top_t);
	mexPrintf("- Samples:1e%d ",(int)log10(NumSample));
	mexPrintf("- Budget:1e%d ",(int)log10(budget));
	mexPrintf("......\n");mexEvalString("drawnow");
	// compute the extension for matrices
	//-------------------------------------
	// Compute weight
	//-------------------------------------
	double SumofW = 0;
	double *weight = (double*)malloc(rankSizeExt*sizeof(double));
	memset(weight, 0, rankSizeExt*sizeof(double));
	start = clock();
	for (uint m = 0; m < rankSize; ++m){
		for (uint n = 0; n < rankSize; ++n){
			for (uint h = 0; h < rankSize; ++h){
					double p = 0.0;
					for(uint i = 0; i < Arow; ++i){
						p += abs(MatA(i,m)*MatA(i,n)*MatA(i,h));
					}
					// extension for matrix B
					double q = 0.0;
					for(uint j = 0; j < Brow; ++j){
						q += abs(MatB(j,m)*MatB(j,n)*MatB(j,h));
					}
					// extension for matrix C
					size_t r = m*rankSize*rankSize + n*rankSize + h;
					weight[r] = p*q;
					SumofW += p*q; 
			}
		}
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	mexPrintf("%f during the initialization phase.\n",duration);mexEvalString("drawnow");
	//-------------------------
	// Do Sampling
	//-------------------------
	start = clock();
	size_t SumCr = 0;
	size_t *freq_r = (size_t *)malloc(rankSizeExt*sizeof(size_t));
	memset( freq_r, 0, rankSizeExt*sizeof(size_t));
	for (uint r = 0; r < rankSizeExt; ++r){
		double u = (double)rand()/(double)RAND_MAX;
		double c = (double)NumSample*weight[r]/SumofW;
		if(u < (c - floor(c)))
			freq_r[r] = (size_t)ceil(c);
		else
			freq_r[r] = (size_t)floor(c);
		SumCr += freq_r[r];
	}
	
	uint *IdxI = (uint*)malloc(SumCr*sizeof(uint));
	memset(IdxI, 0, SumCr*sizeof(uint));
	uint *IdxJ = (uint*)malloc(SumCr*sizeof(uint));
	memset(IdxJ, 0, SumCr*sizeof(uint));
	// sample indexes
	double *pa = (double *)malloc(Arow*sizeof(double));
	double *pb = (double *)malloc(Brow*sizeof(double));
	memset( pa, 0, Arow*sizeof(double));
	memset( pb, 0, Brow*sizeof(double));
	std::map<point2D, double> IrJc;
	size_t offset = 0;
	for (uint m = 0; m < rankSize; ++m){
		for (uint n = 0; n < rankSize; ++n){
			for (uint h = 0; h < rankSize; ++h){
				double sum_a = 0.0;
				for(uint i = 0; i < Arow; ++i){
					sum_a += abs(MatA(i,m)*MatA(i,n)*MatA(i,h));
					pa[i] = sum_a;
				}
				double sum_b = 0.0;
				for(uint j = 0; j < Brow; ++j){
					sum_b += abs(MatB(j,m)*MatB(j,n)*MatB(j,h));
					pb[j] = sum_b;
				}
				size_t r = m*rankSize*rankSize + n*rankSize + h;
				binary_search(freq_r[r], (IdxI + offset), Arow, pa);
				binary_search(freq_r[r], (IdxJ + offset), Brow, pb);
				offset += freq_r[r];
			}
		}
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	mexPrintf("%f during the sampling phase.\n",duration);mexEvalString("drawnow");
	// compute update value and saved in map<pair, value>
	// use map IrJc to save the sampled values
	offset = 0;
	start = clock();
	for (uint m = 0; m < rankSize; ++m){
		for (uint n = 0; n < rankSize; ++n){
			for (uint h = 0; h < rankSize; ++h){
				size_t r = m*rankSize*rankSize + n*rankSize + h;
				for(size_t s = 0; s < freq_r[r]; ++s){
					uint idxi = IdxI[offset];
					uint idxj = IdxJ[offset];
					double score = sgn(MatA(idxi,m))*sgn(MatA(idxi,n))*sgn(MatA(idxi,h));
					score *= sgn(MatB(idxj,m))*sgn(MatB(idxj,n))*sgn(MatB(idxj,h));
					IrJc[point2D(idxi, idxj)] += score;
					++offset;
				}
			}
		}
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	mexPrintf("%f during the computing score.\n",duration);mexEvalString("drawnow");
	//-----------------------------------
	//sort the values have been sampled
	//-----------------------------------
	// sort by actual value
	std::vector<pidx2d> sortVec;
	start = clock();
	for (auto mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr){
		double true_value = MatrixColMul(mapItr->first, AT, BT);
		sortVec.push_back(std::make_pair(mapItr->first, true_value));
	}
	// sort the vector according to the actual value
	sort(sortVec.begin(), sortVec.end(), compgt<pidx2d>);
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	mexPrintf("%f during the sorting phase.\n",duration);mexEvalString("drawnow");
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
	}
	mexPrintf("Done!\n");
	//---------------
	// free
	//---------------
	free(weight);
	free(IdxI);
	free(IdxJ);
	free(freq_r);
	free(pa);
	free(pb);
}
