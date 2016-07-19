/*
	Equality Sampling with three factor matrices
	usage:
	[value, time, indexes] =  equalitySampling(A, B, C, budget, samples, top_t);

	* Variables input:
		A, B, C: are factor matrices, suppose R is the rank of tensor
				they have the same columns size
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

double getValue(const point3D &coord, \
				   Matrix &A, \
				   Matrix &B, \
				   Matrix &C){
	double ans = 0;
    for (size_t k = 0; k < A.col; ++k){
        ans += A.GetElement(coord.x,k) * \
        	   B.GetElement(coord.y,k) * \
        	   C.GetElement(coord.z,k);
    }
    return ans;
}

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	srand(unsigned(time(NULL)));
	//--------------------
	// Initialization
	//--------------------
	size_t rankSize = mxGetN(prhs[0]);
	Matrix MatA(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
	Matrix MatB(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
	Matrix MatC(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));
	// the budget
	size_t leNbuget = mxGetN(prhs[3]);
	size_t* budget = (size_t*)malloc(leNbuget*sizeof(size_t));
	for(size_t i = 0; i  < leNbuget; ++i){
		budget[i] = (size_t)mxGetPr(prhs[3])[i];
	}
	// number of samples
	const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
	// find the top-t largest value
	const size_t top_t = (size_t)mxGetPr(prhs[5])[0];
	// topValue
	const double topValue = mxGetPr(prhs[6])[0];

	// recall of different budget
	plhs[0] = mxCreateDoubleMatrix(leNbuget, 1, mxREAL);
	double *recall = mxGetPr(plhs[0]);
	//-------------------------------------
	// Compute weight
	//-------------------------------------
	double SumofW = 0;
	double *weight = (double*)malloc(rankSize*sizeof(double));
	memset(weight, 0, rankSize*sizeof(double));
	double tempW = 0;
	for (int r = 0; r < rankSize; ++r){
		weight[r] = MatA.SumofCol[r];
		weight[r] *= MatA.SumofCol[r];
		weight[r] *= MatA.SumofCol[r];
		SumofW += weight[r]; 
	}

	//-------------------------
	// Do Sampling
	//-------------------------
	size_t *freq_r = (size_t *)malloc(rankSize*sizeof(size_t));
	memset( freq_r, 0, rankSize*sizeof(size_t));
	double u = 0.0;
	double c = 0.0;
	for (size_t r = 0; r < rankSize; ++r){
		u = (double)rand()/(double)RAND_MAX;
		c = (double)NumSample*weight[r]/SumofW;
		if(u < (c - floor(c)))
			freq_r[r] = ceil(c);
		else
			freq_r[r] = floor(c);
	}
	size_t *IdxI = (size_t*)malloc((NumSample + rankSize)*sizeof(size_t));
	memset(IdxI, 0, (NumSample + rankSize)*sizeof(size_t));
	size_t *IdxJ = (size_t*)malloc((NumSample + rankSize)*sizeof(size_t));
	memset(IdxJ, 0, (NumSample + rankSize)*sizeof(size_t));
	size_t *IdxK = (size_t*)malloc((NumSample + rankSize)*sizeof(size_t));
	memset(IdxK, 0, (NumSample + rankSize)*sizeof(size_t));
	size_t *IdxR = (size_t*)malloc((NumSample + rankSize)*sizeof(size_t));
	memset(IdxR, 0, (NumSample + rankSize)*sizeof(size_t));
	// sample indexes
	size_t offset = 0;
	for (int r = 0; r < rankSize; ++r){
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
	size_t idxi, idxj, idxk;
	// use map IrJc to save the sampled values
	std::map<point3D, double> IrJc;
	offset = 0;
	for(int r = 0; r < rankSize; ++r){
		for(int s = 0; s < freq_r[r]; ++s){
			idxi = IdxI[offset];
			idxj = IdxJ[offset];
			idxk = IdxK[offset];
			IrJc[point3D(idxi, idxj, idxk)] += 1.0;
			++offset;
		}
	}
	//-----------------------------------
	//sort the values have been sampled
	//-----------------------------------
	// for pre sort
	std::vector<indValue> tempSortedVec;
	// push the value into a vector for sorting
	std::map<point3D, double>::iterator mapItr;
	for (mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr){
		tempSortedVec.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	sort(tempSortedVec.begin(), tempSortedVec.end(), cmp);
	// diffrernt budget
	for (size_t s = 0; s < leNbuget; ++s){
		
		std::vector<indValue> sortVec;
		double true_value = 0;
		// compute the top-t' (budget) actual value
		for(size_t m = 0; m < tempSortedVec.size() && m < budget[s]; ++m){
			true_value = getValue(tempSortedVec[m].first, MatA, MatB, MatC);
			sortVec.push_back(std::make_pair(tempSortedVec[m].first,true_value));
		}
		// sort the vector according to the actual value
		sort(sortVec.begin(), sortVec.end(), cmp);
		double recall_temp = 0.0;
		for(size_t t = 0; t < top_t; ++t){
			if(sortVec[t].second >= topValue)
				recall_temp += 1;
		}
		recall[s] = recall_temp/top_t;
	}
	//---------------
	// free
	//---------------
	free(IdxI);
	free(IdxJ);
	free(IdxK);
	free(IdxR);
	free(freq_r);
}