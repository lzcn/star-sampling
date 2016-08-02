/*
	[sampled_value, actual_vaule] = equalityTRIAL003(A, B, C, samples, top_t)
*/

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
	Matrix MatA(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
	Matrix MatB(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
	Matrix MatC(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));
	const size_t NumSample = (size_t)mxGetPr(prhs[3])[0];
	
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
	
	plhs[3] = mxCreateDoubleMatrix(1, 1, mxREAL);
	mxGetPr(plhs[3])[0] = SumofW;
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
	size_t *IdxRp = (size_t*)malloc((NumSample + rankSize)*sizeof(size_t));
	memset(IdxRp, 0, (NumSample + rankSize)*sizeof(size_t));
	size_t *IdxRpp = (size_t*)malloc((NumSample + rankSize)*sizeof(size_t));
	memset(IdxRpp, 0, (NumSample + rankSize)*sizeof(size_t));
	size_t *IdxRppp = (size_t*)malloc((NumSample + rankSize)*sizeof(size_t));
	memset(IdxRppp, 0, (NumSample + rankSize)*sizeof(size_t));
	vose_alias( NumSample + rankSize, IdxRp, rankSize, weight, SumofW);
	vose_alias( NumSample + rankSize, IdxRpp, rankSize, weight, SumofW);
	vose_alias( NumSample + rankSize, IdxRppp, rankSize, weight, SumofW);
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
	size_t idxi, idxj, idxk, idxrp, idxrpp, idxrppp;
	// use map IrJc to save the sampled values
	std::map<point3D, double> IrJc;
	double valueSampled = 1.0;
	offset = 0;
	for(int r = 0; r < rankSize; ++r){
		for(int s = 0; s < freq_r[r]; ++s){
			idxi = IdxI[offset];
			idxj = IdxJ[offset];
			idxk = IdxK[offset];
			idxrp = IdxRp[offset];
			idxrpp = IdxRpp[offset];
			idxrppp = IdxRppp[offset];
			valueSampled = 1.0;
			valueSampled *= sgn_foo(MatA.GetElement(idxi,r));
			valueSampled *= sgn_foo(MatB.GetElement(idxj,r));
			valueSampled *= sgn_foo(MatC.GetElement(idxk,r));
			valueSampled *= MatA.GetElement(idxi,idxrp)/MatA.SumofCol[idxrp];
			valueSampled *= MatB.GetElement(idxj,idxrp)/MatB.SumofCol[idxrp];
			valueSampled *= MatC.GetElement(idxk,idxrp)/MatC.SumofCol[idxrp];
			IrJc[point3D(idxi, idxj, idxk)] += valueSampled;
			valueSampled *= MatA.GetElement(idxi,idxrp)/MatA.SumofCol[idxrp];
			valueSampled *= MatB.GetElement(idxj,idxrp)/MatB.SumofCol[idxrp];
			valueSampled *= MatC.GetElement(idxk,idxrp)/MatC.SumofCol[idxrp];
			valueSampled *= SumofW;
			valueSampled *= MatA.GetElement(idxi,idxrpp)/MatA.SumofCol[idxrpp];
			valueSampled *= MatB.GetElement(idxj,idxrpp)/MatB.SumofCol[idxrpp];
			valueSampled *= MatC.GetElement(idxk,idxrpp)/MatC.SumofCol[idxrpp];
			valueSampled *= SumofW;
			valueSampled *= MatA.GetElement(idxi,idxrppp)/MatA.SumofCol[idxrppp];
			valueSampled *= MatB.GetElement(idxj,idxrppp)/MatB.SumofCol[idxrppp];
			valueSampled *= MatC.GetElement(idxk,idxrppp)/MatC.SumofCol[idxrppp];
			valueSampled *= SumofW;		
			++offset;
		}
	}

	//-----------------------------------
	//sort the values have been sampled
	//-----------------------------------
	std::vector<std::pair<double,double> > cmpvalue;
	std::map<point3D, double>::iterator mapItr;
	double true_value = 0;
	for (mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr){
		true_value = getValue(mapItr->first, MatA, MatB, MatC);
		cmpvalue.push_back(std::make_pair(mapItr->second,true_value));
	}
	plhs[0] = mxCreateDoubleMatrix(cmpvalue.size(), 1, mxREAL);
	double *score = mxGetPr(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(cmpvalue.size(), 1, mxREAL);
	double *actual_vaule = mxGetPr(plhs[1]);

	//--------------------------------
	// Converting to Matlab
	//--------------------------------
	for(size_t m = 0; m < cmpvalue.size(); ++m){
		score[m] = cmpvalue[m].first*SumofW*SumofW/NumSample;
		actual_vaule[m] = cmpvalue[m].second*cmpvalue[m].second;
	}
	printf("%f\n",SumofW);
	
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
}
