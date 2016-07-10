/*
	Diamond Sampling with N factor matrices
	It will return a sparse tensor stored 
	the sampled result
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

/*
	suppose the dimension of feature is d;
	The first matrix has d rows;
	The left matrices have d columns;

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
	Matrix MatC(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));
	const int top_t = (int)mxGetPr(prhs[3])[0];
	const size_t NumSample = (size_t)mxGetPr(prhs[4])[0];
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
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
			tempW *= MatB.SumofCol[k];
			tempW *= MatC.SumofCol[k];
			weight[k*MatA.col + i] = tempW;
			SumofW += tempW;
		}
	}
	printf("%f\n", SumofW);
	for (size_t k = 0; k < MatA.row; ++k){
		for(size_t i = 0; i < MatA.col; ++i){
			weight[k*MatA.col + i] /= SumofW;
		}
	}
	SumofW = 1.0;
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during computing weight\n",duration);

	//-------------------------
	// Do Sampling
	//-------------------------

	start = clock();
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
	// sampled k's frequency 
	size_t *freq_k = (size_t*)malloc(MatA.row*sizeof(size_t));
	memset(freq_k, 0, MatA.row*sizeof(size_t));

	// Do sample S pairs (k, i) ,
	sample_index(NumSample, WeightInd, \
				 IndforM, IndforK, \
				 freq_k, \
				 MatA.row, MatA.col, \
				 weight, SumofW);
	// sample k';
	for (int s = 0; s < NumSample; ++s){
		IndforKp[s] = MatA.randRow(IndforM[s]);
	}
	// sample n;
	size_t offset = 0;
	for (int k = 0; k < MatA.row; ++k){
		vose_alias( freq_k[k], (IndforN + offset), \
					MatB.row, \
					(MatB.element + k*MatB.row), \
					MatB.SumofCol[k]);
		offset += freq_k[k];
	}
	
	// sample p;
	offset = 0;
	for (int k = 0; k < MatA.row; ++k){
		vose_alias( freq_k[k], (IndforP + offset), \
					MatC.row, \
					(MatC.element + k*MatC.row), \
					MatC.SumofCol[k]);
		offset += freq_k[k];
	}
	// compute update value and saved in map<pair, value>
	double valueSampled = 1.0;
	size_t idxm, idxn, idxp, idxk, idkp;
	// use map IrJc to save the sampled values
	std::map<point3D, double> IrJc;
	for (int s = 0; s < NumSample ; ++s){
		idxk = IndforK[s];
		idkp = IndforKp[s];
		idxm = IndforM[s];
		idxn = IndforN[s];
		idxp = IndforP[s];
		valueSampled = 1.0;
		valueSampled *= sgn_foo(MatA.GetElement(idxk,idxm));
		valueSampled *= sgn_foo(MatB.GetElement(idxn,idxk));
		valueSampled *= sgn_foo(MatC.GetElement(idxp,idxk));
		valueSampled *= sgn_foo(MatA.GetElement(idkp,idxm));
		valueSampled *= MatB.GetElement(idxn,idkp);
		valueSampled *= MatC.GetElement(idxp,idkp);
		// Update the element in coordinate
		IrJc[point3D(idxm,idxn,idxp)] += valueSampled;
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during sampling\n",duration);

	//-----------------------------------
	//sort the values have been sampled
	//-----------------------------------

	start = clock();
	std::vector<indValue> sortVec;
	std::map<point3D, double>::iterator mapItr;
	double true_value = 0;

	for (mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr){
		true_value = vectors_mul(mapItr->first, MatA, MatB, MatC);
		sortVec.push_back(std::make_pair(mapItr->first,true_value));
		//sortVec.push_back(std::make_pair(mapItr->first,mapItr->second));
	}
	sort(sortVec.begin(),sortVec.end(),cmp);

	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during computer and sorting tensor \n",duration);
 
	//--------------------------------
	// Converting to Matlab
	//--------------------------------
	start = clock();
	size_t phls_row = sortVec.size();
	// pair
	plhs[0] = mxCreateNumericMatrix(phls_row, 3, mxUINT64_CLASS, mxREAL);
	uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[0]);
	// value
	plhs[1] = mxCreateDoubleMatrix(phls_row, 1, mxREAL);
	double *plhs_result = mxGetPr(plhs[1]);
	for(size_t m = 0; m < sortVec.size(); ++m){
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
	printf("%f seconds during converting \n",duration);
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
	free(freq_k);

}