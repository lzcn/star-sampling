#include <map>
#include <set>
#include <cstdio>
#include <ctime>
#include "dmatrix.h"
#include "util.h"
#include "coordinate.h"
#include "random.h"
#include "mex.h"

double diff_y_pred(DMatrix &U,DMatrix &I,DMatrix &T,uint u,uint i,uint pos,uint neg){
	uint rank = U.row;
	double ans = 0.0;
	for(uint r = 0; r < rank; ++r){
		ans += U(r,u)*I(r,i)*(T(r,pos)-T(r,neg));
	}
	return ans;
}
double Frobenius(DMatrix &A){
	double ans = 0.0;
	for(uint i = 0; i < A.col*A.row; i++){
		ans += A.value[i]*A.value[i];
	}
	return (ans);
}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
    //--------------------------------------------
    // Remove some tuple untill to be 5-core data
    //--------------------------------------------
	clock_t start;
	uint NumofEntry = mxGetM(prhs[0]);
	double *userid = mxGetPr(prhs[0]);
	double *itemid = mxGetPr(prhs[1]);
    double *tagid  = mxGetPr(prhs[2]);
    // record the data to a map, and count user,item,tag
	std::set<point3D> UserItemTag;
	std::map<uint, int> UserCount;
	std::map<uint, int> ItemCount;
	std::map<uint, int> TagCount;
	for(auto i = 0; i < NumofEntry; ++i){
        UserItemTag.insert( point3D(userid[i],itemid[i],tagid[i]) );
        UserCount[(uint)userid[i]] += 1;
        ItemCount[(uint)itemid[i]] += 1;
        TagCount[ (uint)tagid[i] ] += 1;
	}
	// do delete
	for(;;){
		std::set<point3D> temp;
        temp = UserItemTag;
		bool changed = false;
		mexPrintf("Current length of tuple:%d\n", temp.size());mexEvalString("drawnow");
		for(auto itr = temp.begin(); itr != temp.end(); ++itr){
			if(UserCount[itr->x] < 5 || ItemCount[itr->y] < 5 || TagCount[itr->z] < 5){
				// delete the tuple
				UserItemTag.erase(*itr);
				// decrement count for <user, movie, tag>
				--UserCount[itr->x];
				--ItemCount[itr->y];
				--TagCount[itr->z];
				// switch the state to changed
				changed = true;
			}
		}
		// if element changed in the loop, then complete
		if(changed) continue;
		else break;
	}
    // comprese the range of user, item, tag from 0 to the size;
	// compression map
	std::map<uint,uint> UserMap;
	std::map<uint,uint> ItemMap;
	std::map<uint,uint> TagMap;
	// find the ID has occurred
	for(auto itr = UserItemTag.begin(); itr != UserItemTag.end(); ++itr){
		UserMap[itr->x] = 1;
		ItemMap[itr->y] = 1;
		TagMap[itr->z] = 1;
	}
	// record the compression map
	uint offset = 0;
	for(auto itr = UserMap.begin(); itr != UserMap.end(); ++itr){
		itr->second = offset++;
	}
	offset = 0;
	for(auto itr = ItemMap.begin(); itr != ItemMap.end(); ++itr){
		itr->second = offset++;
	}
	offset = 0;
	for(auto itr = TagMap.begin(); itr != TagMap.end(); ++itr){
		itr->second = offset++;
	}
    // map to save <posts, posTag>
	typedef std::set<uint> posTag;
    std::map<point2D, posTag> Posts;
	
	for(auto itr = UserItemTag.begin(); itr != UserItemTag.end(); ++itr){
        uint user = UserMap[itr->x];
        uint item = ItemMap[itr->y];
        uint tag  = TagMap[itr->z];
        Posts[point2D(user,item)].insert(tag);
	}
	mexPrintf("Posts size :%d\n",Posts.size());mexEvalString("drawnow");
    //////////////////////////////
    // SGD for factorization
    //////////////////////////////
    // initialized the factor matrices
    uint userSize = UserMap.size();
    uint itemSize = ItemMap.size();
    uint tagSize = TagMap.size();
    uint factorSize = 8;
    double mean = 0.25;
    double stdev = 0.1;
    double lambda = 1e-6;
    double alpha = 0.1;
    uint maxiter = 20;
	mexPrintf("SGD for factorization ......\n");mexEvalString("drawnow");
	// postive tag index for each post
	uint *t_pos_Ind = (uint*)malloc(Posts.size()*tagSize*sizeof(uint));
	// negative tag index for each post
	uint *t_neg_Ind = (uint*)malloc(Posts.size()*tagSize*sizeof(uint));
	uint *uInd = (uint*)malloc(Posts.size()*sizeof(uint));
	uint *iInd = (uint*)malloc(Posts.size()*sizeof(uint));
	double *ptSize = (double*)malloc(Posts.size()*sizeof(double));
	offset = 0;
	for(auto itr = Posts.begin(); itr != Posts.end(); ++itr){
		uInd[offset] = itr->first.x;
		iInd[offset] = itr->first.y;
		ptSize[offset] = (itr->second).size();
		for(uint t = 0; t < tagSize; ++t){
			uint m = 0;
			uint n = 0;
			// if postive tag
			if( (itr->second).end() != (itr->second).find(t) ){
				t_pos_Ind[offset*tagSize + m] = t;
				++m;
			}else{
				t_neg_Ind[offset*tagSize + n] = t;
				++n;
			}
		}
		++offset;
	}
	// initialize U,I,T
	plhs[0] = mxCreateDoubleMatrix(factorSize, userSize, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(factorSize, itemSize, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(factorSize, tagSize, mxREAL);
    DMatrix MatUser(factorSize, userSize);
    MatUser.init(INIT_RAND_N,mean,stdev);
    DMatrix MatItem(factorSize, itemSize);
    MatItem.init(INIT_RAND_N, mean, stdev);
    DMatrix MatTag(factorSize, tagSize);
    MatTag.init(INIT_RAND_N, mean, stdev);
	// SGD method to update factor matrices
	double *inter_v = (double*)malloc(factorSize*sizeof(double));
	double *inter_q = (double*)malloc(factorSize*sizeof(double));
	// w(t+,t-) = s(y(u,i,t+,t-))[1- s(y(u,i,t+,t-))]
	double *wtptn = (double*)malloc(tagSize*tagSize*sizeof(double));
	memset(wtptn, 0, tagSize*tagSize*sizeof(double));
    for(uint l = 0; l < maxiter;++l){
		//for (u,i) in Posts do
		start = clock();
		// objective function
		double AUC = 0.0;
		AUC -= Frobenius(MatUser);
		AUC -= Frobenius(MatItem);
		AUC -= Frobenius(MatTag);
		AUC *= lambda;
		for(uint s = 0; s < Posts.size(); ++s){
			// post_auc = (1/z) * \sum{t+}\sum{t-}w(t+,t-)s(y(u,i,t+,t-))
			double post_auc = 0.0;			
			uint u = uInd[s];
			uint i = iInd[s];
			uint psize = ptSize[s];
			uint nsize = tagSize - psize;
			double z = (double)( psize * nsize );
			// compute the wtptn before each iteration
			// wtptn is computed is first, so the old value will be updated
			memset(inter_v, 0, factorSize*sizeof(double));
			for(uint m = 0; m < psize; ++m){
				uint tag_pos = t_pos_Ind[s*tagSize + m];
				for(uint n = 0; n < nsize; ++n){
					uint tag_neg = t_neg_Ind[s*tagSize + n];
					double y = diff_y_pred(MatUser, MatItem, MatTag, u, i, tag_pos, tag_neg);
					double wtptn_ui = sigmoid(y)*(1 - sigmoid(y));
					wtptn[tag_pos * tagSize + tag_neg] = wtptn_ui;
					post_auc += wtptn_ui*sigmoid(y);
					for(uint f = 0; f < factorSize; ++f){
						inter_v[f] += wtptn_ui*(MatTag(f,tag_pos) - MatTag(f,tag_neg));
					}
				}
			}
			AUC += post_auc/z;
			//---------------------------------
            // update user and item matrices
			//---------------------------------
            for(uint f = 0; f < factorSize; ++f){
                MatUser(f,u) += alpha*(MatItem(f,i)*inter_v[f]/z - lambda*MatUser(f,u));
                MatItem(f,i) += alpha*(MatUser(f,u)*inter_v[f]/z - lambda*MatItem(f,i));
            }
			// update tag matrix
			// compute the intermediate variables
			for(uint f = 0; f < factorSize; ++f){
				inter_q[f] = MatUser(f,u)*MatItem(f,i);
			}
			//----------------------------
			// update the postive tag
			//----------------------------
			// grad(t+) = (1/z)*\sum{t-}w(t+,t-)*inter_q;
			for(uint m = 0; m < psize; ++m){
				uint tag_pos = t_pos_Ind[s*tagSize + m];
				double gradt = 0.0;
				for(uint n = 0; n < nsize; ++n){
					uint tag_neg = t_neg_Ind[s*tagSize + n];
					gradt += wtptn[tag_pos * tagSize + tag_neg];
				}
				for(uint f = 0; f < factorSize; ++f){
					MatTag(f,tag_pos)  += alpha*(-inter_q[f]*gradt/z - lambda*MatTag(f,tag_pos));
				}
			}
			//----------------------------
			// update the negative tag
			//----------------------------
			for(uint n = 0; n < nsize; ++n){
				uint tag_neg = t_neg_Ind[s*tagSize + n];
				double gradt = 0.0;
				for(uint m = 0; m < psize; ++m){
					uint tag_pos = t_pos_Ind[s*tagSize + m];
					gradt += wtptn[tag_pos * tagSize + tag_neg];
				}
				for(uint f = 0; f < factorSize; ++f){
					MatTag(f,tag_neg)  += alpha*(inter_q[f]*gradt/z - lambda*MatTag(f,tag_neg));
				}
			}
        }
		mexPrintf("Iter:%d,ObjectFuntcionValue:%f, TimeDuration:%f\n",l,AUC,timeDuration(start));mexEvalString("drawnow");
    }
	///////////////////
	// Conveter
	//////////////////
	for(uint i = 0; i < userSize*factorSize; ++i){
		mxGetPr(plhs[0])[i] = MatUser.value[i];
	}
	for(uint i = 0; i < itemSize*factorSize; ++i){
		mxGetPr(plhs[1])[i] = MatItem.value[i];
	}
	for(uint i = 0; i < userSize*factorSize; ++i){
		mxGetPr(plhs[2])[i] = MatTag.value[i];
	}
	//--------------
	// free
	//--------------
	free(inter_q);
	free(inter_v);
	free(wtptn);
	free(uInd);
	free(iInd);
	free(t_pos_Ind);
	free(t_neg_Ind);
	free(ptSize);
}
