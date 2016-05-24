#include "mex.h"
#include <cstdio>
#include <cmath>
#include <vector>
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
	size_t m1 = mxGetM(prhs[0]);
	printf("%d\n", m1);
	size_t m2 = mxGetM(prhs[3]);
	printf("%d\n", m2);
	size_t row = 3;
	double *userIDa = mxGetPr(prhs[0]);
	double *movieIDa = mxGetPr(prhs[1]);
	double *rating = mxGetPr(prhs[2]);
	double *userIDb = mxGetPr(prhs[3]);
	double *movieIDb = mxGetPr(prhs[4]);
	double *tagID = mxGetPr(prhs[5]);
	std::vector<size_t> v[3];
	std::vector<double> value;
	size_t index_start = 0;
	for(size_t i = 0; i < m2; ++i){
		//printf("%f\n", userIDb[i]);
		for(size_t j = 0; j < m1; ++j){
			if(abs(userIDb[i] - userIDa[j]) < 1e-6 && abs(movieIDb[i] - movieIDa[j]) < 1e-6){
				index_start = j;
				v[0].push_back(static_cast<size_t>(userIDb[i]));
				v[1].push_back(static_cast<size_t>(movieIDb[i]));
				v[2].push_back(static_cast<size_t>(tagID[i]));
				value.push_back(rating[j]);
			}
		}
	}
	int num = value.size();
	plhs[0] = mxCreateNumericMatrix(num, 3, mxUINT64_CLASS, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(num, 1, mxREAL);
	uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[0]);
	double *plhs_val = mxGetPr(plhs[1]);
	FILE* fout = fopen("out_tmp.dat","w");
	fprintf(fout, "userID\titemID\ttagID\tposts\n");
	for (int i = 0; i < num; ++i)
	{	
		plhs_pr[i] = v[0][i];
		plhs_pr[num + i] = v[1][i];
		plhs_pr[num*2 + i] = v[2][i];
		plhs_val[i] = value[i];
		fprintf(fout, "%d\t%d\t%d\t%.1f\n",v[0][i],v[1][i],v[2][i],value[i]);
	}
	fclose(fout);
}