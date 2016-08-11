/*
	Diamond Sapmling with N factor matrixes
	It will return a sparse tansor stored 
	the sampled result
*/

#include <vector>
#include <map>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <ctime>
#include "mex.h"
#include "matrix.h"

void mexFunction (size_t nlhs, mxArray *plhs[], size_t nrhs, const mxArray *prhs[])
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
	Matrix MatC(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));
	
	const size_t budget = (size_t)mxGetPr(prhs[3])[0];
	const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
	const size_t top_t = (size_t)mxGetPr(prhs[5])[0];
	// output
	plhs[0] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
	double *plhs_result = mxGetPr(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	double *tsec = mxGetPr(plhs[1]);
	plhs[2] = mxCreateNumericMatrix(top_t, 3, mxUINT64_CLASS, mxREAL);
	uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[2]);	
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec = duration;
	printf("%f seconds during initialization\n",duration);

	//-------------------------------------
	// Compute weight
	//-------------------------------------

	start = clock();
	double SumofW = 0;
	//weight has the same size of A
	double *weight = (double*)malloc(MatA.row*MatA.col*sizeof(double));
	memset(weight, 0, MatA.row*MatA.col*sizeof(double));
	double tempW = 0;
	// weight[r * MatA.col + i] : i-th column r-th row
	for (size_t r = 0; r < MatA.row; ++r){
		for(size_t i = 0; i < MatA.col; ++i){
			//w_{ri} = |a_{ri}|*||a_{*i}||_1*||b_{*r}||_1
			tempW = 1;
			tempW *= abs(MatA.GetElement(r,i));
			tempW *= MatB.SumofCol[r];
			tempW *= MatC.SumofCol[r];
			weight[r*MatA.col + i] = tempW;
			SumofW += tempW;
		}
	}
	for (size_t r = 0; r < MatA.row; ++r){
		for(size_t i = 0; i < MatA.col; ++i){
			weight[r*MatA.col + i] /= SumofW;
		}
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	printf("%f seconds during computing weight\n",duration);

	//-------------------------
	// Do Sampling
	//-------------------------

	// sampled index  for weight
	size_t *WeightInd = (size_t *)malloc(NumSample*sizeof(size_t));
	memset(WeightInd, 0, NumSample*sizeof(size_t));
	// sampled r, i, j, k
	size_t *IndforI = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforI, 0, NumSample*sizeof(size_t));
	size_t *IndforJ = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforJ, 0, NumSample*sizeof(size_t));
	size_t *IndforK = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforK, 0, NumSample*sizeof(size_t));
	size_t *IndforR = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforR, 0, NumSample*sizeof(size_t));	
	// sampled r's frequency
	size_t *freq_r = (size_t*)malloc(MatA.row*sizeof(size_t));
	memset(freq_r, 0, MatA.row*sizeof(size_t));
	
	start = clock();
	// Do sample S pairs (k, i) ,
	sample_index(NumSample, WeightInd, \
				 			 IndforI, IndforR, \
				 			 freq_r, \
				 			 MatA.row, MatA.col, \
				 			 weight, 1.0);

	// sample j and k;
	size_t offset = 0;
	for (size_t r = 0; r < MatA.row; ++r){
		vose_alias( freq_r[r], (IndforJ + offset), \
								MatB.row, \
								(MatB.element + r*MatB.row), \
								MatB.SumofCol[r]);
		vose_alias( freq_r[r], (IndforK + offset), \
								MatC.row, \
								(MatC.element + r*MatC.row), \
								MatC.SumofCol[r]);
		offset += freq_r[r];
	}
	// compute update value and saved in map<pair, value>
	double valueSampled = 1.0;
	std::map<point3D, double> IrJc;
	for (size_t s = 0; s < NumSample ; ++s){
		size_t r = IndforR[s];
		size_t i = IndforI[s];
		size_t j = IndforJ[s];
		size_t k = IndforK[s];
		valueSampled = 1.0;
		valueSampled *= sgn_foo(MatA.GetElement(r,i));
		valueSampled *= sgn_foo(MatB.GetElement(j,r));
		valueSampled *= sgn_foo(MatC.GetElement(k,r));
		IrJc[point3D(i,j,k)] += valueSampled;
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during sampling\n",duration);

	//-----------------------------------
	//sort the values have been sampled
	//-----------------------------------

	std::vector<pidx3d> tempSortedVec;
	std::vector<pidx3d> sortVec;
	std::map<point3D, double>::iterator mapItr;
	for (mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr){
		tempSortedVec.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	start = clock();
	sort(tempSortedVec.begin(), tempSortedVec.end(), compgt<pidx3d>);
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	mexPrintf("%f seconds during pre-sorting\n",duration);

	start = clock();
	double true_value = 0;
	for(size_t m = 0; m < tempSortedVec.size() && m < budget; ++m){
		true_value = vectors_mul(tempSortedVec[m].first, MatA, MatB, MatC);
		sortVec.push_back(std::make_pair(tempSortedVec[m].first,true_value));
	}
	sort(sortVec.begin(), sortVec.end(), compgt<pidx3d>);

	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	mexPrintf("%f seconds during computing and sorting\n",duration);
 
	//--------------------------------
	// Converting to Matlab
	//--------------------------------
	start = clock();
	for(size_t m = 0; m < sortVec.size() && m < top_t; ++m){
		//value
		plhs_result[m] = sortVec[m].second;
		//i
		plhs_pr[m] = (sortVec[m].first.x + 1);
		//j
		plhs_pr[m + top_t] = (sortVec[m].first.y + 1);
		//k
		plhs_pr[m + top_t + top_t] = (sortVec[m].first.z + 1);
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	mexPrintf("%f seconds during converting \n",duration);
	
	//---------------
	// free
	//---------------
	free(weight);
	free(WeightInd);
	free(IndforI);
	free(IndforJ);
	free(IndforR);
	free(freq_r);

}
