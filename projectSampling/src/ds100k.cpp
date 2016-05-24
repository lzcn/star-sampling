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
#include "tensor.hpp"

typedef std::vector<size_t> coordinate;
typedef std::pair<coordinate,double> PAIR;

int cmp(const PAIR &x,const PAIR&y){
	return x.second > y.second;
}

int sgn_foo(double x){
	return x<0? -1:1;
}
/*
	give an coord(i_0,i_1,i_2,...,i_(N-1))
	conpute the value of U_{(i_0,i_1,i_2,...,i_(N-1))}
*/
double vectors_mul(const coordinate &coord,\
 				std::vector<FactorMatrix*> &vMatrixs,\
 				size_t nrhs,size_t rank_size){
    double ans = 0;
    double* temp = new double[rank_size];
    /*initialize to the column of vMatrixs[0] */
    for (size_t i = 0; i < rank_size; ++i){
        temp[i] = vMatrixs[0]->get_element(i,coord[0]);
    }
    for (size_t i = 1; i < nrhs; ++i){
        for(size_t j = 0; j < rank_size; ++j){
            temp[j] *= vMatrixs[i]->get_element(coord[i],j);
        }
    }
    for (size_t i = 0; i < rank_size; ++i){
        ans += temp[i];
    }
    delete []temp;
    return ans;
}

void display(std::vector<size_t> v){
	std::vector<size_t>::iterator itr;
	for (itr = v.begin(); itr != v.end(); ++itr){
		printf("%d, ",(*itr));
	}
}
void display_map(std::map<std::vector<size_t>,double> v){
	std::map<std::vector<size_t>,double> ::const_iterator itr;
	for (itr = v.begin(); itr != v.end(); ++itr){
		display(itr->first);
		printf("%f\n",itr->second);
	}
}

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	clock_t start,finish;
	double duration;
	srand(unsigned(time(NULL)));
	/* number of samples */
	const size_t num_sample = 100000;
	/* initialization */
	start = clock();
	std::vector<FactorMatrix*> vMatrixs;
	for (int i = 0; i < nrhs; ++i){
		vMatrixs.push_back(new FactorMatrix(prhs[i]));
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during initialization\n",duration);	
	start = clock();
	/* start with matrix:prhs[0] */
	size_t RANK,L_HEAD;
	RANK = mxGetM(prhs[0]);
	L_HEAD = mxGetN(prhs[0]);
	/* step 1: Get the weight matrix*/
	double sum_wpr = 0;
	double *weight = new double[RANK*L_HEAD];
	double w;
	start = clock();
	for (size_t k = 0; k < RANK; ++k){
		for(size_t i = 0; i < L_HEAD; ++i){
			w = 1;
			w *= abs(vMatrixs[0]->get_element(k,i));
			w *= vMatrixs[0]->sum_of_col()[i];
			for (size_t j = 1; j < nrhs; ++j){
				w *= vMatrixs[j]->sum_of_col()[k];
			}
			weight[i+k*L_HEAD] = w;
			sum_wpr += w;
		}
	}

	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during computing weight\n",duration);
	/* Tensor to save samping result */
	//std::map<coordinate,double> Tensor;
	std::map<coordinate,double> Tensor;
	coordinate coords[num_sample];

	start = clock();
	size_t *index_weight = new size_t[num_sample];
	std::map<size_t, size_t> freq_k;
	sample_index(num_sample, index_weight, L_HEAD, freq_k, RANK*L_HEAD, weight, sum_wpr);
	//vose_alias(num_sample, index_weight, M*N, weight, sum_wpr);
	size_t index, i_0, i_n, k, kp;
	int sgn = 1;
	double value_of_sampled;
	// push back all i_0
	for (int s = 0; s < num_sample; ++s){
		index = index_weight[s];
		i_0 = index % L_HEAD;		
		coords[s].push_back(i_0);
	}
	// push back left i_n
	size_t *sampled_in = new size_t[num_sample];
	for (int i = 1; i < nrhs; ++i){
		size_t sum_k = 0;
		for (int k = 0; k < RANK; ++k){
			size_t n = vMatrixs[i]->get_row();
			double sum_pdf = vMatrixs[i]->sum_of_col()[k];
			size_t *start_result = sampled_in + sum_k;
			double *start_pdf = vMatrixs[i]->get_pr() + k*n;
			vose_alias(freq_k[k],start_result,n,start_pdf,sum_pdf);
			sum_k += freq_k[k];
		}
		for (int s = 0; s < num_sample; ++s){
			coords[s].push_back(sampled_in[s]);
		}
	}
	delete []sampled_in;
	//compute x
	for (int s = 0; s < num_sample ; ++s){
		/* Step 2: Sample coordinate */
		sgn = 1;
		/* sample i_0 and k with weight matrix*/
		index = index_weight[s];
		/*(i_0,k) sampled*/
		i_0 = index % L_HEAD;
		k = index / L_HEAD;
		sgn *= sgn_foo(vMatrixs[0]->get_element(k,i_0));
		for(size_t i_n = 1; i_n < nrhs; ++i_n){ 
			sgn *= sgn_foo(vMatrixs[i_n]->get_element(coords[s][i_n],k));
		}
		/* sample kp from the start matrix*/
		kp = vMatrixs[0]->sample_row_index(i_0);
		sgn *= sgn_foo(vMatrixs[0]->get_element(i_0,kp));
		value_of_sampled = 1.0;
		for (int i = 1; i < nrhs; ++i){
			value_of_sampled *= vMatrixs[i]->get_element(coords[s][i],kp);
		}
		value_of_sampled *= sgn;
		/* Step 3: Update the element in coordinate(i_0,i_1,...,i_N) */
		Tensor[ coords[s] ] += value_of_sampled;
	}
	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during sampling\n",duration);
	/* sort value in tensor */
	start = clock();
	std::vector<PAIR> pair_vec;
	std::map<coordinate,double>::iterator map_itr;
	double true_value = 0;

	for (map_itr = Tensor.begin(); map_itr != Tensor.end(); ++map_itr){

		true_value = vectors_mul((map_itr->first),vMatrixs,nrhs,RANK);
		pair_vec.push_back(make_pair(map_itr->first,true_value));
		//pair_vec.push_back(make_pair(map_itr->first,map_itr->second));
	}
	sort(pair_vec.begin(),pair_vec.end(),cmp);


	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during computer and sorting tenosor \n",duration);

	start = clock();
	size_t phls_row = pair_vec.size();
	plhs[0] = mxCreateNumericMatrix(phls_row, nrhs, mxUINT64_CLASS, mxREAL);
	uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(phls_row, 1, mxREAL);
	double *plhs_result = mxGetPr(plhs[1]);
	for(size_t m = 0; m < pair_vec.size(); ++m){
		plhs_result[m] = pair_vec[m].second;
		//plhs_result[m] = vectors_mul(pair_vec[m].first,vMatrixs,nrhs,RANK);
		for(size_t n = 0; n < nrhs; ++n){
			/*index start with 1*/
			plhs_pr[m+n*phls_row] = ((pair_vec[m].first[n])+1);
		}
	}

	finish = clock();
	duration = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("%f seconds during converting \n",duration);

	//display_map(Tensor);
	delete []weight;
	delete []index_weight;
	std::vector<FactorMatrix*>::iterator itr;
	for (itr = vMatrixs.begin(); itr != vMatrixs.end() ; ++itr){
		delete *itr;
	}
}
