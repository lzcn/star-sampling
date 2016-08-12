/*
	Wedge Sampling for matrix multiplication
	Author: Zhi Lu
	References:	[1] "Diamond Sampling for Approximate Maximum 
					All-pairs Dot-product(MAD) Search";
				[2] "Approximation Matrix Multiplication 
					for Pattern Recognition Tasks"
*/
/*
	Diamond Sampling for matrix multiplication
	Author: Zhi Lu
	Reference:"Diamond Sampling for Approximate Maximum 
			All-pairs Dot-product(MAD) Search"
*/
#include <utility>
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
	start = clock();
	Matrix MatA(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
	Matrix MatB(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
	const size_t budget = (size_t)mxGetPr(prhs[2])[0];
	const size_t NumSample = (size_t)mxGetPr(prhs[3])[0];
	const size_t top_t = (size_t)mxGetPr(prhs[4])[0]
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	mexPrintf("Starting Wedge Sampling:");
	mexPrintf("- Top-%d ",top_t);
	mexPrintf("- Samples:%d ",NumSample);
	mexPrintf("- Budget:%d ",budget);
	mexPrintf("......");
	//-------------------------------------
	// Compute weight
	//-------------------------------------
	double SumofW = 0;
	//weight has the same size of A
	double *weight = (double*)malloc(MatA.row*MatA.col*sizeof(double));
	memset(weight, 0, MatA.row*MatA.col*sizeof(double));
	start = clock();
	for (size_t r = 0; r < MatA.row; ++r){
		for(size_t i = 0; i < MatA.col; ++i){
			double tempW = abs(MatA.GetElement(r,i)) * MatB.SumofCol[r];
			weight[r*MatA.col + i] = tempW;
			SumofW += tempW;
		}
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	//-------------------------
	// Do Sampling
	//-------------------------
	start = clock();
	// sampled r, i, j, r'
	size_t *IdxR = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IdxR, 0, NumSample*sizeof(size_t));	
	size_t *IdxI = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IdxI, 0, NumSample*sizeof(size_t));
	size_t *IdxJ = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IdxJ, 0, NumSample*sizeof(size_t));
	// sampled r's frequency 
	size_t *freq_r = (size_t*)malloc(MatA.row*sizeof(size_t));
	memset(freq_r, 0, MatA.row*sizeof(size_t));
	// sample pairs (i,r) ,
	binary_sample(NumSample, \
				 IdxI, IdxR, \
				 freq_r, \
				 MatA.row, MatA.col, \
				 weight, SumofW);
	// sample j;
	for (size_t r = 0,offset = 0; r < MatA.row; ++r){
		vose_alias( freq_r[r], (IdxJ + offset), \
					MatB.row, \
					(MatB.element + r*MatB.row), \
					MatB.SumofCol[r]);
		offset += freq_r[r];
	}
	// sample rp and  get score
	std::map<point2D, double> IrJc;
	for (size_t s = 0; s < NumSample ; ++s){
		size_t r = IdxR[s];
		size_t i = IdxI[s];
		size_t j = IdxJ[s];
		double valueSampled = sgn_foo(MatA.GetElement(r,i)) \
		 					* sgn_foo(MatB.GetElement(j,r));
		// Update the element in coordinate
		IrJc[point2D(i,j)] += valueSampled;
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	
	//-----------------------------------
	//sort the values have been sampled
	//-----------------------------------
	std::vector<pidx2d> sortVec;
	std::vector<pidx2d> tempSortedVec;
	std::map<point2D, double>::iterator mapItr;
	// sort the sampled value
	for (mapItr = IrJc.begin(); mapItr != IrJc.end() ; ++mapItr){
		tempSortedVec.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	start = clock();
	sort(tempSortedVec.begin(), tempSortedVec.end(), compgt<pidx2d>);
	finish = clock();
	*tsec += duration;
	// compute the actual of top-t'(budget)
	for(size_t m = 0; m < tempSortedVec.size() && m < budget; ++m){
		double true_value = vectors_mul(tempSortedVec[m].first, MatA, MatB);
		sortVec.push_back(std::make_pair(tempSortedVec[m].first,true_value));
	}
	sort(sortVec.begin(), sortVec.end(), compgt<pidx2d>);
	finish = clock();
 	duration = (double)(finish-start) / CLOCKS_PER_SEC;
 	*tsec += duration;	
	//--------------------------------
	// Converting to Matlab
	//--------------------------------
	for(size_t m = 0; m < sortVec.size() && m < top_t; ++m){
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
	free(IdxR);
	free(IdxRp);
	free(freq_r);

}
