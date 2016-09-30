#include <vector>
#include <map>
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
	double duration;
	srand(unsigned(time(NULL)));
	//--------------------
	// Initialization
	//--------------------
	uint rankSize = (uint)mxGetN(prhs[0]);
	uint rankSizeExt = rankSize * rankSize;
	// original matrices
	double *A = mxGetPr(prhs[0]);
	double *B = mxGetPr(prhs[1]);
	double *C = mxGetPr(prhs[2]);
	uint Arow = mxGetM(prhs[0]);
	uint Brow = mxGetM(prhs[1]);
	uint Crow = mxGetM(prhs[2]);
	// the budget
	const size_t budget = (size_t)mxGetPr(prhs[3])[0];
	// number of samples
	const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
	// find the top-t largest value
	const uint top_t = (uint)mxGetPr(prhs[5])[0];
	// result of sampling
	plhs[0] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
	double *values = mxGetPr(plhs[0]);
	// time duration sampling
	plhs[1] = mxCreateDoubleMatrix(1, 4, mxREAL);
	double *tsec = mxGetPr(plhs[1]);
	// indexes of values
	plhs[2] = mxCreateNumericMatrix(top_t, 3, mxUINT64_CLASS, mxREAL);
	uint64_T* indexes = (uint64_T*)mxGetData(plhs[2]);
	mexPrintf("Starting Core^2 Sampling:");
	mexPrintf("Top:%d,Samples:1e%d,Budget:1e%d\n",top_t,(int)log10(NumSample),(int)log10(budget));
	mexEvalString("drawnow");
	// compute the extension for matrices
	Timer sTime;
	start = clock();
	Matrix AT(rankSize,Arow);
	Matrix BT(rankSize,Brow);
	Matrix CT(rankSize,Crow);
	AT.transpose(mxGetPr(prhs[0]));
	BT.transpose(mxGetPr(prhs[1]));
	CT.transpose(mxGetPr(prhs[2]));
	double *Aex = (double*)malloc(mxGetM(prhs[0])*rankSizeExt*sizeof(double));
	double *Bex = (double*)malloc(mxGetM(prhs[1])*rankSizeExt*sizeof(double));
	double *Cex = (double*)malloc(mxGetM(prhs[2])*rankSizeExt*sizeof(double));
	memset(Aex, 0, mxGetM(prhs[0])*rankSizeExt*sizeof(double));
	memset(Bex, 0, mxGetM(prhs[1])*rankSizeExt*sizeof(double));
	memset(Cex, 0, mxGetM(prhs[2])*rankSizeExt*sizeof(double));
	// compute the extension matrices
	for (uint m = 0; m < rankSize; ++m){
		for (uint n = 0; n < rankSize; ++n){
			// extension for matrix A
			size_t r = m * rankSize + n;
			double sum = 0;
			for(uint i = 0; i < Arow; ++i){
				sum += abs(A[m * Arow + i] * A[n * Arow + i]);
				Aex[r * Arow + i] = sum;
			}
			// extension for matrix B
			sum = 0;
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
	Matrix MatAex(mxGetM(prhs[0]), rankSizeExt, Aex, MATRIX_NONE_SUM);
	Matrix MatBex(mxGetM(prhs[1]), rankSizeExt, Bex, MATRIX_NONE_SUM);
	Matrix MatCex(mxGetM(prhs[2]), rankSizeExt, Cex, MATRIX_NONE_SUM);
	sTime.r_init(start);
	//-------------------------------------
	// Compute weight
	//-------------------------------------
	double SumofW = 0;
	double *weight = (double*)malloc(rankSizeExt*sizeof(double));
	memset(weight, 0, rankSizeExt*sizeof(double));
	start = clock();
	for (uint r = 0; r < rankSizeExt; ++r){
		weight[r] = MatAex(Arow - 1, r);
		weight[r] *= MatBex(Brow - 1, r);
		weight[r] *= MatCex(Crow - 1, r);
		SumofW += weight[r]; 
	}
	sTime.r_init(start);
	mexPrintf("|-%f during the initialization phase.\n",sTime.initialization);
	mexEvalString("drawnow");
	//-------------------------
	// Do Sampling
	//-------------------------
	start = clock();
	size_t TotalS = 0;
	size_t *freq_r = (size_t *)malloc(rankSizeExt*sizeof(size_t));
	memset( freq_r, 0, rankSizeExt*sizeof(size_t));
	for (uint r = 0; r < rankSizeExt; ++r){
		double u = (double)rand()/(double)RAND_MAX;
		double c = (double)NumSample*weight[r]/SumofW;
		if(u < (c - floor(c)))
			freq_r[r] = (size_t)ceil(c);
		else
			freq_r[r] = (size_t)floor(c);
		TotalS += freq_r[r];
	}
	uint *IdxI = (uint*)malloc(TotalS*sizeof(uint));
	memset(IdxI, 0, TotalS*sizeof(uint));
	uint *IdxJ = (uint*)malloc(TotalS*sizeof(uint));
	memset(IdxJ, 0, TotalS*sizeof(uint));
	uint *IdxK = (uint*)malloc(TotalS*sizeof(uint));
	memset(IdxK, 0, TotalS*sizeof(uint));
	// sample indexes
	size_t offset = 0;
	for (uint r = 0; r < rankSizeExt; ++r){
		binary_search( freq_r[r], (IdxI + offset), Arow, (MatAex.element + r*Arow));
		binary_search( freq_r[r], (IdxJ + offset), Brow, (MatBex.element + r*Brow));
		binary_search( freq_r[r], (IdxK + offset), Crow, (MatCex.element + r*Crow));
		offset += freq_r[r];
	}
	sTime.r_samp(start);
	mexPrintf("|-%f during the sampling phase.\n",sTime.sampling);
	mexEvalString("drawnow");
	// compute update value and saved in map<pair, value>
	// use map IrJc to save the sampled values
	start = clock();
	// std::map<point3D, double> IrJc;
	TPoint3DMap IrJc;
	if(budget >= NumSample){
		for(size_t i = 0; i < TotalS; ++i){
			IrJc[point3D(IdxI[i], IdxJ[i], IdxK[i])] = 1;
		}
	}else{
		offset = 0;
		for (uint m = 0; m < rankSize; ++m){
			for (uint n = 0; n < rankSize; ++n){
				size_t r = m*rankSize + n;
				for(size_t s = 0; s < freq_r[r]; ++s,++offset){
					uint idxi = IdxI[offset];
					uint idxj = IdxJ[offset];
					uint idxk = IdxK[offset];
					double value = sgn(AT(m,idxi)) * sgn(AT(n,idxi));
					value *= sgn(BT(m,idxj)) * sgn(BT(n,idxj));
					value *= sgn(CT(m,idxk)) * sgn(CT(n,idxk));
					IrJc[point3D(idxi, idxj, idxk)] += value;
				}
			}
		}
	}
	sTime.r_score(start);
	mexPrintf("|-%f during the scoring phase.\n",sTime.scoring);
	mexEvalString("drawnow");
	//-----------------------------------
	//sort the values have been sampled
	//-----------------------------------
	// for pre-sort
	start = clock();
	std::vector<pidx3d> tempSortedVec;
	std::vector<pidx3d> sortVec;
	// push the value into a vector for sorting
	for (auto mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr){
		tempSortedVec.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	sort(tempSortedVec.begin(), tempSortedVec.end(), compgt<pidx3d>);
	for(size_t m = 0; m < tempSortedVec.size() && m < budget; ++m){
		double true_value = MatrixColMul(tempSortedVec[m].first, AT, BT, CT);
		sortVec.push_back(std::make_pair(tempSortedVec[m].first,true_value));
	}
	// sort the vector according to the actual value
	sort(sortVec.begin(), sortVec.end(), compgt<pidx3d>);
	sTime.r_filter(start);
 	mexPrintf("|-%f during the sorting phase.\n",sTime.filtering);
	mexEvalString("drawnow");
	//--------------------------------
	// Converting to Matlab
	//--------------------------------
	// value
	for(size_t m = 0; m < sortVec.size() && m < top_t; ++m){
		//value
		values[m] = sortVec[m].second;
		//indexes
		indexes[m] = (sortVec[m].first.x + 1);
		indexes[m + top_t] = (sortVec[m].first.y + 1);
		indexes[m + top_t + top_t] = (sortVec[m].first.z + 1);
	}
	tsec[0] = sTime.initialization;
	tsec[1] = sTime.sampling;
	tsec[2] = sTime.scoring;
	tsec[3] = sTime.filtering;
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
