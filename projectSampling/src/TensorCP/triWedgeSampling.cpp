/*
	Diamond Sampling with Three factor matrices 
	usage:
	[value, time, indexes] =  triWedgeSampling(A, B, C, budget, samples, top_t);

	* Variables input:
		A, B, C: are factor matrices, suppose R is the rank of tensor
				A has R rows, B, C have R columns
		budget: use top-t' scores to sort
		samples: numbers of samples
		top_t : find the top_t value in tensor

	* Variables output:
		value: the top_t value
		time: time consuming during the sampling
		indexes: the indexes of the corresponding value
		Author : Zhi Lu
*/

#include <vector>
#include <map>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <ctime>

#include "mex.h"
#include "../../include/matrix.h"

typedef std::pair<point3D,double> indValue;

int cmp(const indValue &x,const indValue&y){
	return (x.second > y.second);
}


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
	Matrix MatC(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));
	const size_t budget = (size_t)mxGetPr(prhs[3])[0];
	const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
	const size_t top_t = (size_t)mxGetPr(prhs[5])[0];
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	double *tsec = mxGetPr(plhs[1]);
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
	// weight[k * MatA.col + i] : i-th column k-th row
	for (size_t k = 0; k < MatA.row; ++k){
		for(size_t i = 0; i < MatA.col; ++i){
			//w_{ki} = |a_{ki}|*||a_{*i}||_1*||b_{*k}||_1
			tempW = 1;
			tempW *= abs(MatA.GetElement(k,i));
			tempW *= MatA.SumofCol[i];
			tempW *= MatA.SumofCol[i];
			tempW *= MatB.SumofCol[k];
			tempW *= MatC.SumofCol[k];
			weight[k*MatA.col + i] = tempW;
			SumofW += tempW;
		}
	}
	for (size_t k = 0; k < MatA.row; ++k){
		for(size_t i = 0; i < MatA.col; ++i){
			weight[k*MatA.col + i] /= SumofW;
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
	// sampled k, m, n, p, k'
	size_t *IndforM = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforM, 0, NumSample*sizeof(size_t));
	size_t *IndforN = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforN, 0, NumSample*sizeof(size_t));
	size_t *IndforP = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforP, 0, NumSample*sizeof(size_t));
	size_t *IndforK = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforK, 0, NumSample*sizeof(size_t));	
	size_t *IndforKp = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforKp, 0, NumSample*sizeof(size_t));
	size_t *IndforKpp = (size_t*)malloc(NumSample*sizeof(size_t));
	memset(IndforKpp, 0, NumSample*sizeof(size_t));
	// sampled k's frequency 
	size_t *freq_k = (size_t*)malloc(MatA.row*sizeof(size_t));
	memset(freq_k, 0, MatA.row*sizeof(size_t));

	start = clock();
	// Do sample S pairs (k, i) ,
	sample_index(NumSample, WeightInd, \
				 IndforM, IndforK, \
				 freq_k, \
				 MatA.row, MatA.col, \
				 weight, SumofW);
	// sample k';
	for (int s = 0; s < NumSample; ++s){
	IndforKp[s] = MatA.randRow(IndforM[s]);
		IndforKpp[s] = MatA.randRow(IndforM[s]);
	}
	// sample n;
	size_t offset = 0;
	for (int k = 0; k < MatA.row; ++k){
		vose_alias( freq_k[k], (IndforN + offset), \
					MatB.row, \
					(MatB.element + k*MatB.row), \
					MatB.SumofCol[k]);
		vose_alias( freq_k[k], (IndforP + offset), \
					MatC.row, \
					(MatC.element + k*MatC.row), \
					MatC.SumofCol[k]);		
		offset += freq_k[k];
	}
	// compute update value and saved in map<pair, value>
	double valueSampled = 1.0;
	size_t idxm, idxn, idxp, idxk, idkp, idkpp;
	// use map IrJc to save the sampled values
	std::map<point3D, double> IrJc;
	for (int s = 0; s < NumSample ; ++s){
		idxk = IndforK[s];
		idkp = IndforKp[s];
		idkpp = IndforKpp[s];
		idxm = IndforM[s];
		idxn = IndforN[s];
		idxp = IndforP[s];
		valueSampled = 1.0;
		valueSampled *= sgn_foo(MatA.GetElement(idxk,idxm));
		valueSampled *= sgn_foo(MatB.GetElement(idxn,idxk));
		valueSampled *= sgn_foo(MatC.GetElement(idxp,idxk));
		valueSampled *= sgn_foo(MatA.GetElement(idkp,idxm));
		valueSampled *= sgn_foo(MatA.GetElement(idkpp,idxm));
		valueSampled *= MatB.GetElement(idxn,idkp);
		valueSampled *= MatB.GetElement(idxn,idkpp);
		valueSampled *= MatC.GetElement(idxp,idkp);
		valueSampled *= MatC.GetElement(idxp,idkpp);
		// Update the element in coordinate
		IrJc[point3D(idxm,idxn,idxp)] += valueSampled;
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	printf("%f seconds during sampling\n",duration);

	//-----------------------------------
	//sort the values have been sampled
	//-----------------------------------

	std::vector<indValue> tempSortedVec;
	std::vector<indValue> sortVec;
	std::map<point3D, double>::iterator mapItr;

	start = clock();
	for (mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr){
		tempSortedVec.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	sort(tempSortedVec.begin(), tempSortedVec.end(), cmp);
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
	sort(sortVec.begin(), sortVec.end(), cmp);

	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	mexPrintf("%f seconds during computing and sorting\n",duration);
 
	//--------------------------------
	// Converting to Matlab
	//--------------------------------
	start = clock();
	size_t phls_row = sortVec.size();
	// value
	plhs[0] = mxCreateDoubleMatrix(phls_row, 1, mxREAL);
	double *plhs_result = mxGetPr(plhs[0]);	
	// pair
	plhs[2] = mxCreateNumericMatrix(phls_row, 3, mxUINT64_CLASS, mxREAL);
	uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[2]);

	for(size_t m = 0; m < sortVec.size() && m < top_t; ++m){
		//value
		plhs_result[m] = sortVec[m].second;
		//m
		plhs_pr[m] = (sortVec[m].first.x + 1);
		//n
		plhs_pr[m + phls_row] = (sortVec[m].first.y + 1);
		//p
		plhs_pr[m + phls_row + phls_row] = (sortVec[m].first.z + 1);
	}

	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	mexPrintf("%f seconds during converting \n",duration);
	//---------------
	// free
	//---------------
	free(weight);
	free(WeightInd);
	free(IndforM);
	free(IndforN);
	free(IndforP);
	free(IndforK);
	free(IndforKp);
	free(IndforKpp);
	free(freq_k);

}