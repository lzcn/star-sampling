#include <vector>
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
	uint rankSize = mxGetN(prhs[0]);
	// get factor matrices
	Matrix MatA(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]),MATRIX_COL_SUM);
	Matrix MatB(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]),MATRIX_COL_SUM);
	Matrix AT(mxGetN(prhs[0]),mxGetM(prhs[0]));
	Matrix BT(mxGetN(prhs[1]),mxGetM(prhs[1]));
	for(uint r = 0 ; r < rankSize; ++r){
		for (uint i = 0; i < MatA.row; ++i) {
			AT(r,i) = MatA(i,r);
		}
		for (uint j = 0; j < MatB.row; ++j) {
			BT(r,j) = MatB(j,r);
		}
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	// the budget
	const size_t budget = (size_t)mxGetPr(prhs[2])[0];
	// number of samples
	const size_t NumSample = (size_t)mxGetPr(prhs[3])[0];
	// find the top-t largest value
	const uint top_t = (uint)mxGetPr(prhs[4])[0];
	// result of sampling
	plhs[0] = mxCreateDoubleMatrix(top_t, 1, mxREAL);
	double *values = mxGetPr(plhs[0]);
	// time duration sampling
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	double *tsec = mxGetPr(plhs[1]);
	*tsec = duration;
	// indexes of values
	plhs[2] = mxCreateNumericMatrix(top_t, 2, mxUINT64_CLASS, mxREAL);
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
		weight[r] = MatA.SumofCol[r];
		weight[r] *= MatB.SumofCol[r];
		SumofW += weight[r]; 
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	//-------------------------
	// Do Sampling
	//-------------------------

	start = clock();
	size_t SumCr = 0;
	size_t *freq_r = (size_t *)malloc(rankSize*sizeof(size_t));
	memset( freq_r, 0, rankSize*sizeof(size_t));
	for (uint r = 0; r < rankSize; ++r){
		double u = (double)rand()/(double)RAND_MAX;
		double c = (double)NumSample*weight[r]/SumofW;
		if(u < (c - floor(c)))
			freq_r[r] = (size_t)ceil(c);
		else
			freq_r[r] = (size_t)floor(c);
		SumCr += freq_r[r];
	}
	uint *IdxI = (uint*)malloc(SumCr*sizeof(uint));
	memset(IdxI, 0, SumCr*sizeof(uint));
	uint *IdxJ = (uint*)malloc(SumCr*sizeof(uint));
	memset(IdxJ, 0, SumCr*sizeof(uint));
	double *pdfa = (double*)malloc(MatA.row*sizeof(double));
	double *pdfb = (double*)malloc(MatB.row*sizeof(double));
	uint L_a = MatA.row;
	uint L_b = MatB.row;
	// sample indexes
	size_t offset = 0;
	for (uint r = 0; r < rankSize; ++r){
		// sample i
		double sum = 0;
		for(uint i = 0; i < L_a; i++){
			sum += abs(MatA(i,r));
			pdfa[i] = sum;
		}
		binary_search(freq_r[r], (IdxI + offset), L_a, pdfa);
		// sample j
		sum = 0;
		for(uint i = 0; i < L_b; i++){
			sum += abs(MatB(i,r));
			pdfb[i] = sum;
		}
		binary_search(freq_r[r], (IdxJ + offset), L_b, pdfb);
		offset += freq_r[r];
	}

	// compute update value and saved in map<pair, value>
	// use map IrJc to save the sampled values
	//std::map<point2D, double> IrJc;
	TPoint2DMap IrJc;
	offset = 0;
	for(uint r = 0; r < rankSize; ++r){
		for(size_t s = 0; s < freq_r[r]; ++s){
			uint idxi = IdxI[offset];
			uint idxj = IdxJ[offset];
			IrJc[point2D(idxi, idxj)] += sgn(MatA(idxi,r)) * sgn(MatB(idxj,r));
			++offset;
		}
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	//-----------------------------------
	//sort the values have been sampled
	//-----------------------------------
	// for pre-sort
	std::vector<pidx2d> tempSortedVec;
	// sort by actual value
	std::vector<pidx2d> sortVec;
	// push the value into a vector for sorting
	for (auto mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr){
		tempSortedVec.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	start = clock();
	sort(tempSortedVec.begin(), tempSortedVec.end(), compgt<pidx2d>);
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
	start = clock();
	double true_value = 0;
	// compute the top-t' (budget) actual value
	for(size_t m = 0; m < tempSortedVec.size() && m < budget; ++m){
		true_value = MatrixColMul(tempSortedVec[m].first, AT, BT);
		sortVec.push_back(std::make_pair(tempSortedVec[m].first,true_value));
	}
	// sort the vector according to the actual value
	sort(sortVec.begin(), sortVec.end(), compgt<pidx2d>);
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	*tsec += duration;
 	
	//--------------------------------
	// Converting to Matlab
	//--------------------------------
	// value
	for(size_t m = 0; m < sortVec.size() && m < top_t; ++m){
		//value
		values[m] = sortVec[m].second;
		//i
		indexes[m] = (sortVec[m].first.x + 1);
		//j
		indexes[m + top_t] = (sortVec[m].first.y + 1);
	}
	mexPrintf("Done!\n");
	//---------------
	// free
	//---------------
	free(pdfa);
	free(pdfb);
	free(IdxI);
	free(IdxJ);
	free(freq_r);
	free(weight);
}
