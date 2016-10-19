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
	clock_t start;
	srand(unsigned(time(NULL)));
	//--------------------
	// Initialization
	//--------------------
	uint rankSize = (uint)mxGetN(prhs[0]);
	uint Arow = (uint)mxGetM(prhs[0]);
	uint Brow = (uint)mxGetM(prhs[1]);
	uint Crow = (uint)mxGetM(prhs[2]);
	Timer sTime;
	// get factor matrices
	// the budget
	start = clock();
	Matrix MatA(Arow,rankSize);
	Matrix MatB(Brow,rankSize);
	Matrix MatC(Crow,rankSize);
	Matrix AT(rankSize,Arow);
	Matrix BT(rankSize,Brow);
	Matrix CT(rankSize,Crow);
	MatA.accumulation(mxGetPr(prhs[0]));
	MatB.accumulation(mxGetPr(prhs[1]));
	MatC.accumulation(mxGetPr(prhs[2]));
	AT.transpose(mxGetPr(prhs[0]));
	BT.transpose(mxGetPr(prhs[1]));
	CT.transpose(mxGetPr(prhs[2]));
	sTime.r_init(start);
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
	mexPrintf("Starting Core^1 Sampling:");
	mexPrintf("Top:%d,Samples:1e%d,Budget:1e%d\n",top_t,(int)log10(NumSample),(int)log10(budget));
	mexEvalString("drawnow");
	//-------------------------------------
	// Compute weight
	//-------------------------------------
	start = clock();
	double SumofW = 0;
	double *weight = (double*)malloc(rankSize*sizeof(double));
	memset(weight, 0, rankSize*sizeof(double));
	double tempW = 0;
	for (uint r = 0; r < rankSize; ++r){
		weight[r] = MatA(Arow-1,r);
		weight[r] *= MatB(Brow-1,r);
		weight[r] *= MatC(Crow-1,r);
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
	size_t *freq_r = (size_t *)malloc(rankSize*sizeof(size_t));
	memset( freq_r, 0, rankSize*sizeof(size_t));
	for (uint r = 0; r < rankSize; ++r){
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
	for (uint r = 0; r < rankSize; ++r){
		binary_search(freq_r[r], (IdxI + offset), Arow, (MatA.element + r*Arow));
		binary_search(freq_r[r], (IdxJ + offset), Brow, (MatB.element + r*Brow));
		binary_search(freq_r[r], (IdxK + offset), Crow, (MatC.element + r*Crow));
		offset += freq_r[r];
	}
	sTime.r_samp(start);
	mexPrintf("|-%f during the sampling phase.\n",sTime.sampling);
	mexEvalString("drawnow");
	//------------------------------------------
	// Filtering
	//-----------------------------------------
	start = clock();
	// use map IrJc to save the sampled values
	TPoint3DMap IrJc;
	if(budget >= NumSample){
		for(size_t i = 0; i < TotalS; ++i){
			IrJc[point3D(IdxI[i], IdxJ[i], IdxK[i])] = 1;
		}
	}else{
		offset = 0;
		for(uint r = 0; r < rankSize; ++r){
			for(size_t s = 0; s < freq_r[r]; ++s){
				uint idxi = IdxI[offset];
				uint idxj = IdxJ[offset];
				uint idxk = IdxK[offset];
				IrJc[point3D(idxi, idxj, idxk)] += sgn(MatA(idxi,r)) * sgn(MatB(idxj,r)) * sgn(MatC(idxk,r));
				++offset;
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
	free(IdxI);
	free(IdxJ);
	free(IdxK);
	free(freq_r);
	free(weight);
}
