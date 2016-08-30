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
	uint Crow = (uint)mxGetM(prhs[2]);
	// original matrices
	Matrix MatA(Arow, rankSize, mxGetPr(prhs[0]), MATRIX_NONE_SUM);
	Matrix MatB(Brow, rankSize, mxGetPr(prhs[1]), MATRIX_NONE_SUM);
	Matrix MatC(Crow, rankSize, mxGetPr(prhs[2]), MATRIX_NONE_SUM);
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
	// indexes of values
	plhs[2] = mxCreateNumericMatrix(top_t, 3, mxUINT64_CLASS, mxREAL);
	uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[2]);
	mexPrintf("Starting Extension Sampling:");
	mexPrintf("- Top-%d ",top_t);
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
					double t = 0.0;
					for(uint k = 0; k < Crow; ++k){
						t += abs(MatC(k,m)*MatC(k,n)*MatC(k,h));
					}
					size_t r = m*rankSize*rankSize + n*rankSize + h;
					weight[r] = p*q*t;
					SumofW += p*q*t; 
			}
		}
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec = duration;
	mexPrintf("%f during the initialization phase.\n",duration);mexEvalString("drawnow");
	//-------------------------
	// Do Sampling
	//-------------------------
	start = clock();
	size_t *freq_r = (size_t *)malloc(rankSizeExt*sizeof(size_t));
	memset( freq_r, 0, rankSizeExt*sizeof(size_t));
	for (uint r = 0; r < rankSizeExt; ++r){
		double u = (double)rand()/(double)RAND_MAX;
		double c = (double)NumSample*weight[r]/SumofW;
		if(u < (c - floor(c)))
			freq_r[r] = (size_t)ceil(c);
		else
			freq_r[r] = (size_t)floor(c);
	}
	
	uint *IdxI = (uint*)malloc((NumSample + rankSizeExt)*sizeof(uint));
	memset(IdxI, 0, (NumSample + rankSizeExt)*sizeof(uint));
	uint *IdxJ = (uint*)malloc((NumSample + rankSizeExt)*sizeof(uint));
	memset(IdxJ, 0, (NumSample + rankSizeExt)*sizeof(uint));
	uint *IdxK = (uint*)malloc((NumSample + rankSizeExt)*sizeof(uint));
	memset(IdxK, 0, (NumSample + rankSizeExt)*sizeof(uint));
	// sample indexes
	double *pa = (double *)malloc(MatA.row*sizeof(double));
	double *pb = (double *)malloc(MatB.row*sizeof(double));
	double *pc = (double *)malloc(MatC.row*sizeof(double));
	memset( pa, 0, MatA.row*sizeof(double));
	memset( pb, 0, MatB.row*sizeof(double));
	memset( pc, 0, MatC.row*sizeof(double));
	size_t offset = 0;
	for (uint m = 0; m < rankSize; ++m){
		for (uint n = 0; n < rankSize; ++n){
			for (uint h = 0; h < rankSize; ++h){
				double sum_a = 0.0;
				for(uint i = 0; i <Arow; ++i){
					pa[i] = abs(MatA(i,m)*MatA(i,n)*MatA(i,h));
					sum_a += pa[i];
				}
				double sum_b = 0.0;
				for(uint j = 0; j < Brow; ++j){
					pb[j] = abs(MatB(j,m)*MatB(j,n)*MatB(j,h));
					sum_b += pb[j];
				}
				// extension for matrix C
				double sum_c = 0.0;
				for(uint k = 0; k < Crow; ++k){
					pc[k] = abs(MatC(k,m)*MatC(k,n)*MatC(k,h));
					sum_c += pc[k];
				}
				size_t r = m*rankSize*rankSize + n*rankSize + h;
				vose_alias(freq_r[r], (IdxI + offset), MatA.row, pa, sum_a);
				vose_alias(freq_r[r], (IdxJ + offset), MatB.row, pb, sum_b);
				vose_alias(freq_r[r], (IdxK + offset), MatC.row, pc, sum_c);
				offset += freq_r[r];
			}
		}
	}
	// compute update value and saved in map<pair, value>
	// use map IrJc to save the sampled values
	std::map<point3D, double> IrJc;
	offset = 0;
	for(uint r = 0; r < rankSizeExt; ++r){
		for(size_t s = 0; s < freq_r[r]; ++s){
			uint idxi = IdxI[offset];
			uint idxj = IdxJ[offset];
			uint idxk = IdxK[offset];
			IrJc[point3D(idxi, idxj, idxk)] += 1;
			++offset;
		}
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	mexPrintf("%f during the sampling phase.\n",duration);mexEvalString("drawnow");
	//-----------------------------------
	//sort the values have been sampled
	//-----------------------------------
	// sort by actual value
	std::vector<pidx3d> sortVec;
	start = clock();
	for (auto mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr){
		double true_value = MatrixRowMul(mapItr->first, MatA, MatB, MatC);
		sortVec.push_back(std::make_pair(mapItr->first, true_value));
	}
	// sort the vector according to the actual value
	sort(sortVec.begin(), sortVec.end(), compgt<pidx3d>);
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
	free(pa);
	free(pb);
	free(pc);
}
