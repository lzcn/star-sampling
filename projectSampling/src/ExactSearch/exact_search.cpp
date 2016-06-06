
#include <list>
#include <vector>
#include <algorithm>
#include <cstddef>

#include "mex.h"

/*coordiate to save the result*/

void disp_coord(const size_t* cur_ind,int nrhs){
    for(int i = 0; i < nrhs; ++i){
        printf("%d, ", cur_ind[i]);
    }
    printf("\n");

}

class IndexforLoop
{
public:
    IndexforLoop(size_t *p,int d);
    ~IndexforLoop();
    IndexforLoop& operator++();
    const size_t *get_ind(){return cur_ind;};
    bool isdone();
private:
    size_t *max_ind;
    size_t *cur_ind;
    int dim;
    bool done_flag;
};
IndexforLoop::IndexforLoop(size_t *p,int d){
    done_flag = false;
    dim = d;
    max_ind = new size_t[dim+1];
    cur_ind = new size_t[dim];
    for (int i = 0; i < dim; ++i){
        cur_ind[i] = 0;
        max_ind[i] = p[i];
    }
	cur_ind[dim] = 0;
};
IndexforLoop::~IndexforLoop(){
    delete []max_ind;
    delete []cur_ind;
}
bool IndexforLoop::isdone(){
    return done_flag;
}
IndexforLoop& IndexforLoop::operator++(){
    cur_ind[0]++;
    for (int i = 0 ; i < dim; ++i){
        if(cur_ind[i] == max_ind[i]){
	        cur_ind[i] = 0;
            ++cur_ind[i+1];
        }
    }
	if (cur_ind[dim] == 1){
		done_flag = true;
	}
    return (*this);
}
/* element by element multiplication of cloumns*/
double all_col_mul(const size_t *cur_ind,\
            std::vector<double*> &pr,size_t rank_size,int nrhs){
    double ans = 0;
    double *temp = new double[rank_size];
    /*initialize to 1 */
    for (size_t i = 0; i < rank_size; ++i){
        temp[i] = 1;
    }
    double *p;
    for (int i = 0; i < nrhs; ++i){
        /*element-wise mutiplication*/
        p = (pr[i] + cur_ind[i]*rank_size);
        for(size_t j = 0; j < rank_size; ++j){
            temp[j] *= *(p+j);
        }
    }
    for (size_t i = 0; i < rank_size; ++i){
        ans += temp[i];
    }
    delete []temp;
    return ans;
}


int doInsert(double value,double*toInsert,size_t num_top){
    double front,next;
    for(size_t i = 0; i < num_top; ++i){
        if(value > toInsert[i]){
            // find and insert
            front = toInsert[i];
            toInsert[i] = value;
            // shift the left element
            for(int j = (i + 1); j < num_top; ++j){
                next = toInsert[j];
                toInsert[j] = front;
                front = next;
            }
            return i;
        }
    }
    return num_top;
}


// the matrix have the same row size:rank_size
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   
    
    const size_t rank_size = mxGetM(prhs[0]);
    const int num_top = 1000;
    double temp_value = 0;

    std::vector<double*> v_pr;

    size_t *p = new size_t[nrhs];
    for (int i = 0; i < nrhs; ++i){
        v_pr.push_back(mxGetPr(prhs[i]));
        p[i] = mxGetN(prhs[i]);
    }
    IndexforLoop cloop(p,nrhs);
    double init_val = all_col_mul(cloop.get_ind(),v_pr,rank_size,nrhs);
    double *max_value = new double[num_top];
    /*std::vector<double> max_value;
    for(int i = 0; i < num_top;++i){
        max_value.push_back(0);
    }
    */

    double max_v = 0.0;
    while(!cloop.isdone()){
        temp_value = all_col_mul(cloop.get_ind(),v_pr,rank_size,nrhs);
        if(temp_value > max_value[num_top]){
            doInsert(temp_value,max_value,num_top);
        }
        ++cloop;
    }
    delete []p;
    plhs[0] = mxCreateDoubleMatrix(num_top,1,mxREAL);
    double *plhs_max;
    plhs_max = mxGetPr(plhs[0]);
    for(size_t i = 0; i < num_top; ++i){
        plhs_max[i] = max_value[i];
    }
    delete []max_value;

}