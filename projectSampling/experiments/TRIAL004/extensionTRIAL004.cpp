#include <vector>
#include <map>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <ctime>

#include "mex.h"
#include "matrix.h"

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
	size_t rankSizeExt = rankSize * rankSize;
	// normal matrix
	double *A = mxGetPr(prhs[0]);
	double *B = mxGetPr(prhs[1]);
	double *C = mxGetPr(prhs[2]);
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
	
	Matrix MatA(mxGetM(prhs[0]), rankSize, A);
	Matrix MatB(mxGetM(prhs[1]), rankSize, B);
	Matrix MatC(mxGetM(prhs[2]), rankSize, C);
	// extension for matrices
	double *Aex = (double*)malloc(mxGetM(prhs[0])*rankSizeExt*sizeof(double));
	double *Bex = (double*)malloc(mxGetM(prhs[1])*rankSizeExt*sizeof(double));
	double *Cex = (double*)malloc(mxGetM(prhs[2])*rankSizeExt*sizeof(double));
	memset(Aex, 0, mxGetM(prhs[0])*rankSizeExt*sizeof(double));
	memset(Bex, 0, mxGetM(prhs[1])*rankSizeExt*sizeof(double));
	memset(Cex, 0, mxGetM(prhs[2])*rankSizeExt*sizeof(double));
	for (size_t m = 0; m < rankSize; ++m){
		for (size_t n = 0; n < rankSize; ++n){
			// extension for matrix A
			for(size_t i = 0; i < mxGetM(prhs[0]); ++i){
				Aex[(m*rankSize + n) * MatA.row + i] = A[m * MatA.row + i] * A[n * MatA.row + i];
			}
			// extension for matrix B
			for(size_t j = 0; j < mxGetM(prhs[1]); ++j){
				Bex[(m*rankSize + n) * MatB.row + j] = B[m * MatB.row + j] * B[n * MatB.row + j];
			}
			// extension for matrix C
			for(size_t k = 0; k < mxGetM(prhs[2]); ++k){
				Cex[(m*rankSize + n) * MatC.row + k] = C[m * MatC.row + k] * C[n * MatC.row + k];
			}
		}
	}
	// extension matrices
	Matrix MatAex(mxGetM(prhs[0]), rankSizeExt, Aex);
	Matrix MatBex(mxGetM(prhs[1]), rankSizeExt, Bex);
	Matrix MatCex(mxGetM(prhs[2]), rankSizeExt, Cex);

	// recall of different budget
	plhs[0] = mxCreateDoubleMatrix(leNbuget, 1, mxREAL);
	double *recall_v_1 = mxGetPr(plhs[0]);

	//-------------------------------------
	// Compute weight
	//-------------------------------------
	double SumofW = 0;
	double *weight = (double*)malloc(rankSizeExt*sizeof(double));
	memset(weight, 0, rankSizeExt*sizeof(double));
	double tempW = 0;
	for (size_t r = 0; r < rankSizeExt; ++r){
		weight[r] = MatAex.SumofCol[r];
		weight[r] *= MatBex.SumofCol[r];
		weight[r] *= MatCex.SumofCol[r];
		SumofW += weight[r]; 
	}

	//-------------------------
	// Do Sampling
	//-------------------------
	size_t *freq_r = (size_t *)malloc(rankSizeExt*sizeof(size_t));
	memset( freq_r, 0, rankSizeExt*sizeof(size_t));
	double u = 0.0;
	double c = 0.0;
	for (size_t r = 0; r < rankSizeExt; ++r){
		u = (double)rand()/(double)RAND_MAX;
		c = (double)NumSample*weight[r]/SumofW;
		if(u < (c - floor(c)))
			freq_r[r] = ceil(c);
		else
			freq_r[r] = floor(c);
	}
	size_t *IdxI = (size_t*)malloc((NumSample + rankSizeExt)*sizeof(size_t));
	memset(IdxI, 0, (NumSample + rankSizeExt)*sizeof(size_t));
	size_t *IdxJ = (size_t*)malloc((NumSample + rankSizeExt)*sizeof(size_t));
	memset(IdxJ, 0, (NumSample + rankSizeExt)*sizeof(size_t));
	size_t *IdxK = (size_t*)malloc((NumSample + rankSizeExt)*sizeof(size_t));
	memset(IdxK, 0, (NumSample + rankSizeExt)*sizeof(size_t));
	size_t *IdxR = (size_t*)malloc((NumSample + rankSizeExt)*sizeof(size_t));
	memset(IdxR, 0, (NumSample + rankSizeExt)*sizeof(size_t));
	// sample indexes
	size_t offset = 0;
	for (int r = 0; r < rankSizeExt; ++r){
		// sample i
		vose_alias( freq_r[r], (IdxI + offset), \
					MatAex.row, \
					(MatAex.element + r*MatAex.row), \
					MatAex.SumofCol[r]);
		// sample j
		vose_alias( freq_r[r], (IdxJ + offset), \
					MatBex.row, \
					(MatBex.element + r*MatBex.row), \
					MatBex.SumofCol[r]);	
		// sample k
		vose_alias( freq_r[r], (IdxK + offset), \
					MatCex.row, \
					(MatCex.element + r*MatCex.row), \
					MatCex.SumofCol[r]);						
		offset += freq_r[r];
	}
	// compute update value and saved in map<pair, value>
	double valueSampled = 1.0;
	size_t idxi, idxj, idxk;
	// use map IrJc to save the sampled values
	std::map<point3D, double> IrJc;
	offset = 0;
	for(size_t r = 0; r < rankSizeExt; ++r){
		for(size_t s = 0; s < freq_r[r]; ++s){
			idxi = IdxI[offset];
			idxj = IdxJ[offset];
			idxk = IdxK[offset];
			size_t idxr = r / rankSize;
			valueSampled = 1.0;
			valueSampled *= sgn_foo(MatA.GetElement(idxi,idxr));
			valueSampled *= sgn_foo(MatB.GetElement(idxj,idxr));
			valueSampled *= sgn_foo(MatC.GetElement(idxk,idxr));
			IrJc[point3D(idxi, idxj, idxk)] += valueSampled;
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
	// compute the top-t' (budget) actual value
	for (size_t s = 0; s < leNbuget; ++s){
		// sort by actual value
		std::vector<indValue> sortVec;
		for(size_t m = 0; m < tempSortedVec.size() && m < budget[s]; ++m){
			double true_value = getValue(tempSortedVec[m].first, MatA, MatB, MatC);
			sortVec.push_back(std::make_pair(tempSortedVec[m].first, true_value));
		}
		// sort the vector according to the actual value
		sort(sortVec.begin(), sortVec.end(), cmp);
		recall_v_1[s] = 0;
		for(size_t t = 0; t < top_t; ++t){
			if(sortVec[t].second >= topValue)
			recall_v_1[s] += 1;
		}
		recall_v_1[s] /= top_t;
	}
	//---------------
	// free
	//---------------
	free(budget);
	free(weight);
	free(IdxI);
	free(IdxJ);
	free(IdxK);
	free(IdxR);
	free(freq_r);
	free(Aex);
	free(Bex);
	free(Cex);
}