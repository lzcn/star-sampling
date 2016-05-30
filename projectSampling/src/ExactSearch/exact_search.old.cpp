
#include <list>
#include <vector>

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
    size_t *get_ind(){return cur_ind;};
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
    max_ind = new size_t[dim];
    cur_ind = new size_t[dim];
    for (int i = 0; i < dim; ++i){
        cur_ind[i] = 0;
        max_ind[i] = p[i];
    }
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
            if(i == (dim - 1)){
                done_flag = true;
            }
            cur_ind[i] = 0;
            ++cur_ind[i+1];
        }
    }
    return (*this);
}
/* element by element multiplication of cloumns*/
double all_col_mul(const size_t *cur_ind,\
            std::vector<double*> &pr,size_t rank_size,int nrhs){
    double ans = 0;
    double* temp = new double[rank_size];
    /*initialize to 1 */
    for (size_t i = 0; i < rank_size; ++i){
        temp[i] = 1;
    }
    double *p;
    for (size_t i = 0; i < nrhs; ++i){
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
/*
void updatelist(std::list<double> &l,double value){
    if(value > l.front()){
        l.push_front(value);
        l.pop_back();
    }else{
        std::list<double>::iterator itr = l.begin();
        ++itr;
        for(;itr != l.end();++itr){
            if(value > *itr){
                l.insert(itr,value);
                l.pop_back();
            }
        }
    }
}
*/
int do_update(double*p, int num_top, double value){
    if(value < p[(num_top - 1)]) return 0;
    double cur = 0;
    double next = 0;
    int insert_i = 0;
    for(int i = 0; i < num_top; ++i){
        if(value > p[i]){
            insert_i = i;
            break;
        }
    }
    cur = p[insert_i];
    p[insert_i] = value;
    ++insert_i;
    for(int i = insert_i; i < num_top; ++i){
        next  = p[i];
        p[i] = cur;
        cur = next;
    }
    return 1;
}
/*
    all matrix has the same row, which is the rank size
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   
    
    const size_t rank_size = mxGetM(prhs[0]);
    const int num_top = 1000;
    double temp_value = 0;

    //double min_value = 0;
    //double max_value = 0;
    //std::list<double> top1k(1000,0.0);

    //double *top1_value = new double[1000];
    //size_t * top1k_ind = new size_t[1000*nrhs];
    std::vector<double*> v_pr;
    //size_t *max_ind = new size_t[nrhs];
    size_t *p = new size_t[nrhs];
    for (int i = 0; i < nrhs; ++i){
        v_pr.push_back(mxGetPr(prhs[i]));
        p[i] = mxGetN(prhs[i]);
    }
    IndexforLoop cloop(p,nrhs);
    double init_val = all_col_mul(cloop.get_ind(),v_pr,rank_size,nrhs);
    //double *max_value = new double[num_top];
    std::vector<double> max_value;
    for(int i = 0; i < num_top;++i){
        //max_value[i] = init_val;
        max_value.push_back(init_val);
    }
    while(!cloop.isdone()){
        //disp_coord(cloop.get_ind(),nrhs);
        temp_value = all_col_mul(cloop.get_ind(),v_pr,rank_size,nrhs);
        // bigger than the top 1k 
        if(temp_value > max_value[(num_top - 1)]){
            max_value[(num_top - 1)] = temp_value;
            reverse(max_value.begin(),max_value.end());
            //updatelist(top1k,temp_value);
            //do_update(max_value,num_top,temp_value);
            //max_value = max_temp;
            //for (int i = 0; i < nrhs; ++i){
            //   max_ind[i] = cloop.get_ind()[i];
            //}
        }
        ++cloop;
    }
    //plhs[0] = mxCreateNumericMatrix(num_top, nrhs, mxUINT64_CLASS, mxREAL);
    //uint64_T* plhs_pr = (uint64_T*)mxGetData(plhs[0]);

    plhs[0] = mxCreateDoubleMatrix(num_top,1,mxREAL);
    double *plhs_max;
    plhs_max = mxGetPr(plhs[0]);
    //*plhs_max = max_value;
    //std::list<double>::iterator itr = top1k.begin();
    for(size_t i = 0; i < num_top; ++i){
        plhs_max[i] = max_value[i];
    }
    delete []p;
    /*
    for(;itr != top1k.end();++itr){
        plhs_max[i] = *itr;
        ++i;
    }
    */
    /*for(int i = 0; i < 1000; ++i){
        plhs_pr[i] = max_ind[i];
        printf("%d,",max_ind[i]);    
    }
    printf("\nmax value:%f\n", max_value);
    */

    //delete []max_ind;
}