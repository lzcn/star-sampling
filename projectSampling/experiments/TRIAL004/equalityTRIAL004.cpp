
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
	srand(unsigned(time(NULL)));
	//--------------------
	// Initialization
	//--------------------
	uint rankSize = mxGetN(prhs[0]);
	// get factor matrices
	Matrix MatA(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
	Matrix MatB(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
	Matrix MatC(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));
	// the budget
	const uint budget = (uint)mxGetPr(prhs[3])[0];
	// number of samples
	const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
	// find the top-t largest value
	const uint top_t = (uint)mxGetPr(prhs[5])[0];
	// result of sampling
	mexPrintf("Starting Central Sampling:");
	mexPrintf("- Top-%d ",top_t);
	mexPrintf("- Samples:1e%d ",(int)log10(NumSample));
	mexPrintf("- Budget:1e%d ",(int)log10(budget));
	mexPrintf("......");mexEvalString("drawnow");

	// recall of different budget
	plhs[0] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
	double *value_v_1 = mxGetPr(plhs[0]);

	plhs[1] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
	double *value_v_2 = mxGetPr(plhs[1]);

	plhs[2] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
	double *value_v_3 = mxGetPr(plhs[2]);

	plhs[3] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
	double *value_v_4 = mxGetPr(plhs[3]);		
	//-------------------------------------
	// Compute weight
	//-------------------------------------
	double SumofW = 0;
	double *weight = (double*)malloc(rankSize*sizeof(double));
	memset(weight, 0, rankSize*sizeof(double));
	double tempW = 0;
	for (uint r = 0; r < rankSize; ++r){
		weight[r] = MatA.SumofCol[r];
		weight[r] *= MatB.SumofCol[r];
		weight[r] *= MatC.SumofCol[r];
		SumofW += weight[r]; 
	}

	//-------------------------
	// Do Sampling
	//-------------------------
	size_t *freq_r = (size_t *)malloc(rankSize*sizeof(size_t));
	memset( freq_r, 0, rankSize*sizeof(size_t));
	for(uint r = 0; r < rankSize; ++r){
		double u = (double)rand()/(double)RAND_MAX;
		double c = (double)NumSample*weight[r]/SumofW;
		if(u < (c - floor(c)))
			freq_r[r] = (size_t)ceil(c);
		else
			freq_r[r] = (size_t)floor(c);
	}
	uint *IdxI = (uint*)malloc((NumSample + rankSize)*sizeof(uint));
	memset(IdxI, 0, (NumSample + rankSize)*sizeof(uint));
	uint *IdxJ = (uint*)malloc((NumSample + rankSize)*sizeof(uint));
	memset(IdxJ, 0, (NumSample + rankSize)*sizeof(uint));
	uint *IdxK = (uint*)malloc((NumSample + rankSize)*sizeof(uint));
	memset(IdxK, 0, (NumSample + rankSize)*sizeof(uint));
	uint *IdxR = (uint*)malloc((NumSample + rankSize)*sizeof(uint));
	memset(IdxR, 0, (NumSample + rankSize)*sizeof(uint));
	uint *IdxRp = (uint*)malloc((NumSample + rankSize)*sizeof(uint));
	memset(IdxRp, 0, (NumSample + rankSize)*sizeof(uint));
	uint *IdxRpp = (uint*)malloc((NumSample + rankSize)*sizeof(uint));
	memset(IdxRpp, 0, (NumSample + rankSize)*sizeof(uint));
	uint *IdxRppp = (uint*)malloc((NumSample + rankSize)*sizeof(uint));
	memset(IdxRppp, 0, (NumSample + rankSize)*sizeof(uint));
	vose_alias( NumSample + rankSize, IdxRp, rankSize, weight, SumofW);
	vose_alias( NumSample + rankSize, IdxRpp, rankSize, weight, SumofW);
	vose_alias( NumSample + rankSize, IdxRppp, rankSize, weight, SumofW);
	// sample indexes
	size_t offset = 0;
	for (uint r = 0; r < rankSize; ++r){
		// sample i
		vose_alias( freq_r[r], (IdxI + offset), \
					MatA.row, \
					(MatA.element + r*MatA.row), \
					MatA.SumofCol[r]);
		// sample j
		vose_alias( freq_r[r], (IdxJ + offset), \
					MatB.row, \
					(MatB.element + r*MatB.row), \
					MatB.SumofCol[r]);	
		// sample k
		vose_alias( freq_r[r], (IdxK + offset), \
					MatC.row, \
					(MatC.element + r*MatC.row), \
					MatC.SumofCol[r]);						
		offset += freq_r[r];
	}
	// compute update value and saved in map<pair, value>
	uint idxi, idxj, idxk, idxrp, idxrpp, idxrppp;
	// use map IrJc to save the sampled values
	std::map<point3D, double> IrJc_v_1;
	std::map<point3D, double> IrJc_v_2;
	std::map<point3D, double> IrJc_v_3;
	std::map<point3D, double> IrJc_v_4;
	offset = 0;
	double valueSampled = 1.0;
	double sign = 1.0;
	for(uint r = 0; r < rankSize; ++r){
		for(size_t s = 0; s < freq_r[r]; ++s){
			idxi = IdxI[offset];
			idxj = IdxJ[offset];
			idxk = IdxK[offset];
			idxrp = IdxRp[offset];
			idxrpp = IdxRpp[offset];
			idxrppp = IdxRppp[offset];
			sign = 1.0;
			sign *= sgn_foo(MatA.GetElement(idxi,r));
			sign *= sgn_foo(MatB.GetElement(idxj,r));
			sign *= sgn_foo(MatC.GetElement(idxk,r));
			valueSampled = sign;
			IrJc_v_1[point3D(idxi, idxj, idxk)] += valueSampled;

			valueSampled *= MatA.GetElement(idxi,idxrp)/MatA.SumofCol[idxrp];
			valueSampled *= MatB.GetElement(idxj,idxrp)/MatB.SumofCol[idxrp];
			valueSampled *= MatC.GetElement(idxk,idxrp)/MatC.SumofCol[idxrp];
			IrJc_v_2[point3D(idxi, idxj, idxk)] += valueSampled;

			valueSampled *= MatA.GetElement(idxi,idxrpp)/MatA.SumofCol[idxrpp];
			valueSampled *= MatB.GetElement(idxj,idxrpp)/MatB.SumofCol[idxrpp];
			valueSampled *= MatC.GetElement(idxk,idxrpp)/MatC.SumofCol[idxrpp];
			IrJc_v_3[point3D(idxi, idxj, idxk)] += valueSampled;

			valueSampled *= MatA.GetElement(idxi,idxrppp)/MatA.SumofCol[idxrppp];
			valueSampled *= MatB.GetElement(idxj,idxrppp)/MatB.SumofCol[idxrppp];
			valueSampled *= MatC.GetElement(idxk,idxrppp)/MatC.SumofCol[idxrppp];
			IrJc_v_4[point3D(idxi, idxj, idxk)] += valueSampled;
			++offset;
		}
	}
	//-----------------------------------
	//sort the values have been sampled
	//-----------------------------------
	// for pre sort
	std::vector<pidx3d> tempSortedVec_v_1;
	std::vector<pidx3d> tempSortedVec_v_2;
	std::vector<pidx3d> tempSortedVec_v_3;
	std::vector<pidx3d> tempSortedVec_v_4;
	std::vector<pidx3d> sortVec_1;
	std::vector<pidx3d> sortVec_2;
	std::vector<pidx3d> sortVec_3;
	std::vector<pidx3d> sortVec_4;
	// push the value into a vector for sorting
	for(auto mapItr = IrJc_v_1.begin(); mapItr != IrJc_v_1.end(); ++mapItr){
		tempSortedVec_v_1.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	for(auto mapItr = IrJc_v_2.begin(); mapItr != IrJc_v_2.end(); ++mapItr){
		tempSortedVec_v_2.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	for(auto mapItr = IrJc_v_3.begin(); mapItr != IrJc_v_3.end(); ++mapItr){
		tempSortedVec_v_3.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	for(auto mapItr = IrJc_v_4.begin(); mapItr != IrJc_v_4.end(); ++mapItr){
		tempSortedVec_v_4.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	sort(tempSortedVec_v_1.begin(), tempSortedVec_v_1.end(), compgt<pidx3d>);
	sort(tempSortedVec_v_2.begin(), tempSortedVec_v_2.end(), compgt<pidx3d>);
	sort(tempSortedVec_v_3.begin(), tempSortedVec_v_3.end(), compgt<pidx3d>);
	sort(tempSortedVec_v_4.begin(), tempSortedVec_v_4.end(), compgt<pidx3d>);
	// diffrernt budget
	double true_value = 0.0;
	for(size_t m = 0; m < tempSortedVec_v_1.size() && m < budget; ++m){
		true_value = MatrixRowMul(tempSortedVec_v_1[m].first, MatA, MatB, MatC);
		sortVec_1.push_back(std::make_pair(tempSortedVec_v_1[m].first,true_value));
	}
	for(size_t m = 0; m < tempSortedVec_v_2.size() && m < budget; ++m){
		true_value = MatrixRowMul(tempSortedVec_v_2[m].first, MatA, MatB, MatC);
		sortVec_2.push_back(std::make_pair(tempSortedVec_v_2[m].first,true_value));
	}
	for(size_t m = 0; m < tempSortedVec_v_3.size() && m < budget; ++m){
		true_value = MatrixRowMul(tempSortedVec_v_3[m].first, MatA, MatB, MatC);
		sortVec_3.push_back(std::make_pair(tempSortedVec_v_3[m].first,true_value));
	}
	for(size_t m = 0; m < tempSortedVec_v_4.size() && m < budget; ++m){
		true_value = MatrixRowMul(tempSortedVec_v_4[m].first, MatA, MatB, MatC);
		sortVec_4.push_back(std::make_pair(tempSortedVec_v_4[m].first,true_value));
	}
	for(size_t m = 0; m < top_t; ++m){
		value_v_1[m] = sortVec_1[m].second;
		value_v_2[m] = sortVec_2[m].second;
		value_v_3[m] = sortVec_3[m].second;
		value_v_4[m] = sortVec_4[m].second;
		
	}
	//---------------
	// free
	//---------------
	free(IdxI);
	free(IdxJ);
	free(IdxK);
	free(IdxR);
	free(IdxRp);
	free(IdxRpp);
	free(IdxRppp);
	free(freq_r);
	free(weight);
}
