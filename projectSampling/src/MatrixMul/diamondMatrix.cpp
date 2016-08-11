/*
	Diamond Sampling for matrix multiplication
	Author: Zhi Lu
	Reference:"Diamond Sampling for Approximate Maximum 
			All-pairs Dot-product(MAD) Search"
*/
#include <vector>
#include <map>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <ctime>

#include "mex.h"
#include "matrix.h"
/*
	diamond sampling for matrix
	A's size: Rank x M
	B's size: N x Rank
	[value, time] = diamondMatrix(A,B,budget,samples,top_t);
*/
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
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	// get the budget
	const size_t budget = (size_t)mxGetPr(prhs[2])[0];
	// get the  number of samples
	const size_t NumSample = (size_t)mxGetPr(prhs[3])[0];
	// get the top_t
	const size_t top_t = (size_t)mxGetPr(prhs[4])[0];
	// value
	plhs[0] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
	double *plhs_result = mxGetPr(plhs[0]);
	// result for time
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	double *tsec = mxGetPr(plhs[1]);
	*tsec = duration;
	// pair
	plhs[2] = mxCreateNumericMatrix(top_t, 2, mxUINT64_CLASS, mxREAL);
	uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[2]);
	//-------------------------------------
	// Compute weight
	//-------------------------------------

	start = clock();
	double SumofW = 0;
	//weight has the same size of A
	double *weight = (double*)malloc(MatA.row*MatA.col*sizeof(double));
	memset(weight, 0, MatA.row*MatA.col*sizeof(double));
	double tempW = 0;
	// weight[k * MatA.col + i] : i-th column k-th row
	for (size_t k = 0; k < MatA.row; ++k){
		for(size_t i = 0; i < MatA.col; ++i){
			//w_{ki} = |a_{ki}|*||a_{*i}||_1*||b_{*k}||_1
			tempW = 1;
			tempW *= abs(MatA.GetElement(k,i));
			tempW *= MatA.SumofCol[i];
			tempW *= MatB.SumofCol[k];
			weight[k*MatA.col + i] = tempW;
			SumofW += tempW;
		}
	}

	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	printf(">> %f seconds during computing weight\n",duration);

	//-------------------------
	// Do Sampling
	//-------------------------

	start = clock();
	// sampled index  for weight
	size_t *WeightInd = (size_t *)malloc(NumSample*sizeof(size_t));
	memset(WeightInd, 0, NumSample*sizeof(size_t));
	// sampled k, i, j, k'
	size_t *IndforK = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforK, 0, NumSample*sizeof(size_t));	
	size_t *IndforI = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforI, 0, NumSample*sizeof(size_t));
	size_t *IndforJ = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforJ, 0, NumSample*sizeof(size_t));
	size_t *IndforKp = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforKp, 0, NumSample*sizeof(size_t));
	// sampled k's frequency 
	size_t *freq_k = (size_t*)malloc(MatA.row*sizeof(size_t));
	memset(freq_k, 0, MatA.row*sizeof(size_t));
	// Do sample S pairs (k, i) ,
	sample_index(NumSample, WeightInd, \
				 IndforI, IndforK, \
				 freq_k, \
				 MatA.row, MatA.col, \
				 weight, SumofW);
	// sample k';
	for (int s = 0; s < NumSample; ++s){
		IndforKp[s] = MatA.randRow(IndforI[s]);
	}
	// sample j;
	size_t offset = 0;
	for (int k = 0; k < MatA.row; ++k){
		vose_alias( freq_k[k], (IndforJ + offset), \
					MatB.row, \
					(MatB.element + k*MatB.row), \
					MatB.SumofCol[k]);
		offset += freq_k[k];
	}
	// compute update value and saved in map<pair, value>
	double valueSampled = 1.0;
	size_t indi,indj,indk,indkp;
	std::map<point2D, double> IrJc;
	for (int s = 0; s < NumSample ; ++s){
		indk = IndforK[s];
		indkp = IndforKp[s];
		indi = IndforI[s];
		indj = IndforJ[s];
		valueSampled = 1.0;
		valueSampled *= sgn_foo(MatA.GetElement(indk,indi));
		valueSampled *= sgn_foo(MatB.GetElement(indj,indk));
		valueSampled *= sgn_foo(MatA.GetElement(indkp,indi));
		valueSampled *= MatB.GetElement(indj,indkp);
		// Update the element in coordinate
		IrJc[point2D(indi,indj)] += valueSampled;
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	printf(">> %f seconds during sampling\n",duration);

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
	mexPrintf("%f seconds during pre-sorting\n",duration);
	// compute the actual of top-t'(budget)
	double true_value = 0;
	for(size_t m = 0; m < tempSortedVec.size() && m < budget; ++m){
		true_value = vectors_mul(tempSortedVec[m].first, MatA, MatB);
		sortVec.push_back(std::make_pair(tempSortedVec[m].first,true_value));
	}
	sort(sortVec.begin(), sortVec.end(), compgt<pidx2d>);
	finish = clock();
 	duration = (double)(finish-start) / CLOCKS_PER_SEC;
 	*tsec += duration;
 	mexPrintf("%f seconds during computing and sorting\n",duration);
	
	//--------------------------------
	// Converting to Matlab
	//--------------------------------
	start = clock();
	size_t phls_row = sortVec.size();
	for(size_t m = 0; m < sortVec.size() && m < top_t; ++m){
		//value
		plhs_result[m] = sortVec[m].second;
		//i
		plhs_pr[m] = (sortVec[m].first.x + 1);
		//j
		plhs_pr[m + phls_row] = (sortVec[m].first.y + 1);
	}

	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during converting \n",duration);
	
	//---------------
	// free
	//---------------
	free(weight);
	free(WeightInd);
	free(IndforI);
	free(IndforJ);
	free(IndforK);
	free(IndforKp);
	free(freq_k);

}
