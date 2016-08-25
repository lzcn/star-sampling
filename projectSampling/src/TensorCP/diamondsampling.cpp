/*
	Diamond Sampling with Three factor matrices 
	usage:
	[value, time, indexes] =  diamondTensor(A, B, C, budget, samples, top_t);

	* Variables input:
		A:	size: (R, L1)
		B:  size: (L2, R)
		C:  size: (L3, R)
		budget: use top-t' scores to do pre-sorting
		samples: numbers of samples
		top_t : find the top_t value in tensor

		* Variables output:
			value: size: (top_t, 1)
						 the top_t value 
			time: time consuming during the sampling
			indexes: size (top_t, 3)
							 the indexes of the corresponding value	
		Author : Zhi Lu
*/

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
	// input
	start = clock();
	Matrix MatA(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
	Matrix MatB(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
	Matrix MatC(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	const size_t budget = (size_t)mxGetPr(prhs[3])[0];
	const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
	const uint top_t = (uint)mxGetPr(prhs[5])[0];
	// output
	plhs[0] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
	double *plhs_result = mxGetPr(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	double *tsec = mxGetPr(plhs[1]);
	*tsec = duration;
	plhs[2] = mxCreateNumericMatrix(top_t, 3, mxUINT64_CLASS, mxREAL);
	uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[2]);	
	mexPrintf("Starting Dimaond Sampling:");
	mexPrintf("- Top-%d ",top_t);
	mexPrintf("- Samples:1e%d ",(int)log10(NumSample));
	mexPrintf("- Budget:1e%d ",(int)log10(budget));
	mexPrintf("......");mexEvalString("drawnow");
	//-------------------------------------
	// Compute weight
	//-------------------------------------
	double SumofW = 0;
	//weight has the same size of A
	double *weight = (double*)malloc(MatA.row*MatA.col*sizeof(double));
	memset(weight, 0, MatA.row*MatA.col*sizeof(double));
	start = clock();
	for (uint r = 0; r < MatA.row; ++r){
		for(uint i = 0; i < MatA.col; ++i){
			double tempW = abs(MatA.GetElement(r,i));
			tempW *= MatA.SumofCol[i];
			tempW *= MatB.SumofCol[r];
			tempW *= MatC.SumofCol[r];
			weight[r*MatA.col + i] = tempW;
			SumofW += tempW;
		}
	}
	for (uint r = 0; r < MatA.row; ++r){
		for(uint i = 0; i < MatA.col; ++i){
			weight[r*MatA.col + i] /= SumofW;
		}
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	//-------------------------
	// Do Sampling
	//-------------------------
	// sampled r, i, j, k
	uint *IdxI = (uint*)malloc(NumSample*sizeof(uint));
	memset(IdxI, 0, NumSample*sizeof(uint));
	uint *IdxJ = (uint*)malloc(NumSample*sizeof(uint));
	memset(IdxJ, 0, NumSample*sizeof(uint));
	uint *IdxK = (uint*)malloc(NumSample*sizeof(uint));
	memset(IdxK, 0, NumSample*sizeof(uint));
	uint *IdxR = (uint*)malloc(NumSample*sizeof(uint));
	memset(IdxR, 0, NumSample*sizeof(uint));	
	// sampled r's frequency 
	size_t *freq_r = (size_t*)malloc(MatA.row*sizeof(size_t));
	memset(freq_r, 0, MatA.row*sizeof(size_t));

	start = clock();
	// Do sample S pairs (r, i)
	sort_sample(NumSample, \
				 IdxI, IdxR, \
				 freq_r, \
				 MatA.row, MatA.col, \
				 weight, 1.0);
	// sample j,k;
	for (uint r = 0, offset = 0; r < MatA.row; ++r){
		vose_alias( freq_r[r], (IdxJ + offset), \
					MatB.row, \
					(MatB.element + r*MatB.row), \
					MatB.SumofCol[r]);
		vose_alias( freq_r[r], (IdxK + offset), \
					MatC.row, \
					(MatC.element + r*MatC.row), \
					MatC.SumofCol[r]);
		offset += freq_r[r];		
	}
	// compute update value and saved in map<pair, value>
	// use map IrJc to save the sampled values
	std::map<point3D, double> IrJc;
	for (size_t s = 0; s < NumSample ; ++s){
		uint i = IdxI[s];
		uint j = IdxJ[s];
		uint k = IdxK[s];
		uint r = IdxR[s];
		uint rp = MatA.randRow(i);
		double valueSampled = sgn_foo(MatA.GetElement(r,i)) \
							* sgn_foo(MatB.GetElement(j,r)) \
							* sgn_foo(MatC.GetElement(k,r)) \
							* sgn_foo(MatA.GetElement(rp,i)) \
							* MatB.GetElement(j,rp) \
							* MatC.GetElement(k,rp);
		// Update the element in coordinate
		IrJc[point3D(i, j, k)] += valueSampled;
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
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
	start = clock();
	for(uint m = 0; m < tempSortedVec.size() && m < budget; ++m){
		double true_value = vectors_mul(tempSortedVec[m].first, MatA, MatB, MatC);
		sortVec.push_back(std::make_pair(tempSortedVec[m].first,true_value));
	}
	sort(sortVec.begin(), sortVec.end(), compgt<pidx3d>);

	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration; 
	//--------------------------------
	// Converting to Matlab
	//--------------------------------
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
	mexPrintf("Done\n");
	//---------------
	// free
	//---------------
	free(weight);
	free(IdxI);
	free(IdxJ);
	free(IdxK);
	free(IdxR);
	free(freq_r);
}
