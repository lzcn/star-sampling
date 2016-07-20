
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
	double *recall_v_1 = mxGetPr(plhs[0]);

	plhs[1] = mxCreateDoubleMatrix(leNbuget, 1, mxREAL);
	double *recall_v_2 = mxGetPr(plhs[1]);

	plhs[2] = mxCreateDoubleMatrix(leNbuget, 1, mxREAL);
	double *recall_v_3 = mxGetPr(plhs[2]);

	plhs[3] = mxCreateDoubleMatrix(leNbuget, 1, mxREAL);
	double *recall_v_4 = mxGetPr(plhs[3]);		
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
	std::map<point3D, double> IrJc_v_1;
	std::map<point3D, double> IrJc_v_2;
	std::map<point3D, double> IrJc_v_3;
	std::map<point3D, double> IrJc_v_4;
	offset = 0;
	double valueSampled = 1.0;
	double sign = 1.0;
	for(int r = 0; r < rankSize; ++r){
		for(int s = 0; s < freq_r[r]; ++s){
			idxi = IdxI[offset];
			idxj = IdxJ[offset];
			idxk = IdxK[offset];
			idxrp = IdxRp[offset];
			idxrpp = IdxRpp[offset];
			idxrppp = IdxRppp[offset];
			sign = 1.0;
			valueSampled = 1.0;
			sign *= sgn_foo(MatA.GetElement(idxi,r));
			sign *= sgn_foo(MatB.GetElement(idxj,r));
			sign *= sgn_foo(MatC.GetElement(idxk,r));

			IrJc_v_1[point3D(idxi, idxj, idxk)] += valueSampled;

			valueSampled *= MatA.GetElement(idxi,idxrp)/MatA.SumofCol[idxrp];
			valueSampled *= MatB.GetElement(idxj,idxrp)/MatB.SumofCol[idxrp];
			valueSampled *= MatC.GetElement(idxk,idxrp)/MatC.SumofCol[idxrp];
			IrJc_v_2[point3D(idxi, idxj, idxk)] += valueSampled;

			valueSampled *= MatA.GetElement(idxi,idxrpp)/MatA.SumofCol[idxrpp];
			valueSampled *= MatB.GetElement(idxj,idxrpp)/MatB.SumofCol[idxrpp];
			valueSampled *= MatC.GetElement(idxk,idxrpp)/MatC.SumofCol[idxrpp];
			IrJc_v_3[point3D(idxi, idxj, idxk)] += sign*valueSampled;

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
	std::vector<indValue> tempSortedVec_v_1;
	std::vector<indValue> tempSortedVec_v_2;
	std::vector<indValue> tempSortedVec_v_3;
	std::vector<indValue> tempSortedVec_v_4;
	// push the value into a vector for sorting
	std::map<point3D, double>::iterator mapItr;
	for (mapItr = IrJc_v_1.begin(); mapItr != IrJc_v_1.end(); ++mapItr){
		tempSortedVec_v_1.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	for (mapItr = IrJc_v_2.begin(); mapItr != IrJc_v_2.end(); ++mapItr){
		tempSortedVec_v_2.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	for (mapItr = IrJc_v_3.begin(); mapItr != IrJc_v_3.end(); ++mapItr){
		tempSortedVec_v_3.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	for (mapItr = IrJc_v_4.begin(); mapItr != IrJc_v_4.end(); ++mapItr){
		tempSortedVec_v_4.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	sort(tempSortedVec_v_1.begin(), tempSortedVec_v_1.end(), cmp);
	sort(tempSortedVec_v_2.begin(), tempSortedVec_v_2.end(), cmp);
	sort(tempSortedVec_v_3.begin(), tempSortedVec_v_3.end(), cmp);
	sort(tempSortedVec_v_4.begin(), tempSortedVec_v_4.end(), cmp);
	// diffrernt budget
	for (size_t s = 0; s < leNbuget; ++s){
		
		std::vector<indValue> sortVec_v_1;
		std::vector<indValue> sortVec_v_2;
		std::vector<indValue> sortVec_v_3;
		std::vector<indValue> sortVec_v_4;
		double true_value = 0;
		// compute the top-t' (budget) actual value
		for(size_t m = 0; m < tempSortedVec_v_1.size() && m < budget[s]; ++m){
			true_value = getValue(tempSortedVec_v_1[m].first, MatA, MatB, MatC);
			sortVec_v_1.push_back(std::make_pair(tempSortedVec_v_1[m].first,true_value));
		}
		for(size_t m = 0; m < tempSortedVec_v_2.size() && m < budget[s]; ++m){
			true_value = getValue(tempSortedVec_v_2[m].first, MatA, MatB, MatC);
			sortVec_v_2.push_back(std::make_pair(tempSortedVec_v_2[m].first,true_value));
		}
		for(size_t m = 0; m < tempSortedVec_v_3.size() && m < budget[s]; ++m){
			true_value = getValue(tempSortedVec_v_3[m].first, MatA, MatB, MatC);
			sortVec_v_3.push_back(std::make_pair(tempSortedVec_v_3[m].first,true_value));
		}
		for(size_t m = 0; m < tempSortedVec_v_4.size() && m < budget[s]; ++m){
			true_value = getValue(tempSortedVec_v_4[m].first, MatA, MatB, MatC);
			sortVec_v_4.push_back(std::make_pair(tempSortedVec_v_4[m].first,true_value));
		}						
		// sort the vector according to the actual value
		sort(sortVec_v_1.begin(), sortVec_v_1.end(), cmp);
		sort(sortVec_v_2.begin(), sortVec_v_2.end(), cmp);
		sort(sortVec_v_3.begin(), sortVec_v_3.end(), cmp);
		sort(sortVec_v_4.begin(), sortVec_v_4.end(), cmp);

		double recall_temp = 0.0;
		for(size_t t = 0; t < top_t; ++t){
			if(sortVec_v_1[t].second >= topValue)
				recall_temp += 1;
		}
		recall_v_1[s] = recall_temp/top_t;

		recall_temp = 0.0;
		for(size_t t = 0; t < top_t; ++t){
			if(sortVec_v_2[t].second >= topValue)
				recall_temp += 1;
		}
		recall_v_2[s] = recall_temp/top_t;		

		recall_temp = 0.0;
		for(size_t t = 0; t < top_t; ++t){
			if(sortVec_v_3[t].second >= topValue)
				recall_temp += 1;
		}
		recall_v_3[s] = recall_temp/top_t;

		recall_temp = 0.0;
		for(size_t t = 0; t < top_t; ++t){
			if(sortVec_v_4[t].second >= topValue)
				recall_temp += 1;
		}
		recall_v_4[s] = recall_temp/top_t;				
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
}