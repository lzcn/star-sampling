/*
	Diamond Sampling with Three factor matrices 
	usage:
	[value, time, indexes] =  diamondTensor(A, B, C, budget, samples, top_t);

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

/*
	suppose the dimension of feature is d;
	The first matrix has d rows;
	The left matrices have d columns;

*/

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	srand(unsigned(time(NULL)));
	size_t rankSize = mxGetN(prhs[0]);
	Matrix MatA(mxGetM(prhs[0]),mxGetN(prhs[0]),mxGetPr(prhs[0]));
	Matrix MatB(mxGetM(prhs[1]),mxGetN(prhs[1]),mxGetPr(prhs[1]));
	Matrix MatC(mxGetM(prhs[2]),mxGetN(prhs[2]),mxGetPr(prhs[2]));
	// the budget
	size_t leN = mxGetN(prhs[3]);
	size_t* budget = (size_t*)malloc(leN*sizeof(size_t));
	size_t* NumSample = (size_t*)malloc((leN+1)*sizeof(size_t));
	NumSample[0] = 0;
	for(size_t i = 0; i  < leN; ++i){
		budget[i] = (size_t)mxGetPr(prhs[3])[i];
		printf("%d\n", budget[i]);
		NumSample[i+1] = (size_t)mxGetPr(prhs[4])[i];
		printf("%d\n", NumSample[i+1]);
	}
	// find the top-t largest value
	const size_t top_t = (size_t)mxGetPr(prhs[5])[0];
	printf("%d\n",top_t );
	// topValue
	const double topValue = mxGetPr(prhs[6])[0];
	printf("%f\n", topValue);

	// recall of different budget
	plhs[0] = mxCreateDoubleMatrix(leN, 1, mxREAL);
	double *recall = mxGetPr(plhs[0]);
	//-------------------------------------
	// Compute weight
	//-------------------------------------

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
			tempW *= MatA.SumofCol[i];
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

	//-------------------------
	// Do Sampling
	//-------------------------
	// sampled index  for weight
	size_t *WeightInd = (size_t *)malloc(sizeof(size_t));
	size_t *IdxI = (size_t*)malloc(sizeof(size_t));
	size_t *IdxJ = (size_t*)malloc(sizeof(size_t));
	size_t *IdxK = (size_t*)malloc(sizeof(size_t));
	size_t *IdxR = (size_t*)malloc(sizeof(size_t));
	size_t *IdxRp = (size_t*)malloc(sizeof(size_t));
	// sampled r's frequency 
	size_t *freq_r = (size_t*)malloc(MatA.row*sizeof(size_t));
	std::map<point3D, double> IrJc;
	for(size_t ll = 0; ll < leN; ++ll){
		memset(freq_r,0,MatA.row*sizeof(size_t));
		WeightInd = (size_t *)realloc(WeightInd,NumSample[ll+1]*sizeof(size_t));
		memset(WeightInd,0, NumSample[ll+1]*sizeof(size_t));
		IdxI = (size_t*)realloc(IdxI,NumSample[ll+1]*sizeof(size_t));
		memset(IdxI,0, NumSample[ll+1]*sizeof(size_t));
		IdxJ = (size_t*)realloc(IdxJ,NumSample[ll+1]*sizeof(size_t));
		memset(IdxJ,0, NumSample[ll+1]*sizeof(size_t));
		IdxK = (size_t*)realloc(IdxK,NumSample[ll+1]*sizeof(size_t));
		memset(IdxK,0, NumSample[ll+1]*sizeof(size_t));
		IdxR = (size_t*)realloc(IdxR,NumSample[ll+1]*sizeof(size_t));
		memset(IdxR,0, NumSample[ll+1]*sizeof(size_t));
		IdxRp = (size_t*)realloc(IdxRp,NumSample[ll+1]*sizeof(size_t));
		memset(IdxRp,0, NumSample[ll+1]*sizeof(size_t));
		// Do sample S pairs (r, i)
		sample_index(NumSample[ll+1], WeightInd, \
				 	IdxI, IdxR, \
				 	freq_r, \
				 	MatA.row, MatA.col, \
				 	weight, 1.0);
		// sample r';
		for (int s = 0; s < NumSample[ll+1]; ++s){
			IdxRp[s] = MatA.randRow(IdxI[s]);
		}
		// sample n;
		size_t offset = 0;
		for (int r = 0; r < MatA.row; ++r){
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
		double valueSampled = 1.0;
		// use map IrJc to save the sampled values
		for (int s = 0; s < (NumSample[ll+1] - NumSample[ll]) ; ++s){
			size_t i = IdxI[s];
			size_t j = IdxJ[s];
			size_t k = IdxK[s];
			size_t r = IdxR[s];
			size_t rp = IdxRp[s];
			valueSampled = 1.0;
			valueSampled *= sgn_foo(MatA.GetElement(r,i));
			valueSampled *= sgn_foo(MatB.GetElement(j,r));
			valueSampled *= sgn_foo(MatC.GetElement(k,r));
			valueSampled *= sgn_foo(MatA.GetElement(rp,i));
			valueSampled *= MatB.GetElement(j,rp);
			valueSampled *= MatC.GetElement(k,rp);
			// Update the element in coordinate
			IrJc[point3D(i, j, k)] += valueSampled;
		}
		printf("%d\n", IrJc.size());
		//-----------------------------------
		//sort the values have been sampled
		//-----------------------------------
		std::vector<indValue> tempSortedVec;
		std::map<point3D, double>::iterator mapItr;
		for (mapItr = IrJc.begin(); mapItr != IrJc.end(); ++mapItr){
			tempSortedVec.push_back(std::make_pair(mapItr->first,mapItr->second));
		}
		sort(tempSortedVec.begin(), tempSortedVec.end(), cmp);
		double true_value = 0.0;
		std::vector<indValue> sortVec;
		for(size_t m = 0; m < tempSortedVec.size() && m < budget[ll]; ++m){
			true_value = vectors_mul(tempSortedVec[m].first, MatA, MatB, MatC);
			sortVec.push_back(std::make_pair(tempSortedVec[m].first,true_value));
		}
		sort(sortVec.begin(), sortVec.end(), cmp);
		double recall_temp = 0.0;
		for(size_t t = 0; t < top_t; ++t){
			if(sortVec[t].second >= topValue){
				recall_temp = recall_temp + 1.0;
			}
		}
		recall[ll] = recall_temp/((double)top_t);
	}
	//---------------
	// free
	//---------------
	free(weight);
	free(WeightInd);
	free(IdxI);
	free(IdxJ);
	free(IdxK);
	free(IdxR);
	free(IdxRp);
	free(freq_r);
}