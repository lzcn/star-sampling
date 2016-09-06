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
	clock_t start;
	uint NumofEntry = mxGetM(prhs[0]);
	double *userid = mxGetPr(prhs[0]);
	double *itemid = mxGetPr(prhs[1]);
    double *tagid  = mxGetPr(prhs[2]);
	////////////////////////
    // Initialization
	////////////////////////
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
	//--------------------------------------------
	// Remove some tuple untill to be 5-core data
	//--------------------------------------------
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
	/////////////////////////////
	// Range Compression 
	/////////////////////////////
    // comprese the range of user, item, tag from 0 to the size;
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
	uint factorSize = (uint)mxGetPr(prhs[3])[0];
	uint maxiter = (uint)mxGetPr(prhs[4])[0];
    double mean = 0.0;
    double stdev = 0.1;
    double lambda = 1e-6;
    double alpha = 0.1;
	mexPrintf("SGD for factorization ......\n");mexEvalString("drawnow");
	//////////////////////////////
	// initialization for SGD
	/////////////////////////////
	uint *idxtpos = (uint*)malloc(Posts.size()*tagSize*sizeof(uint));
	uint *idxtneg = (uint*)malloc(Posts.size()*tagSize*sizeof(uint));
	uint *idxu = (uint*)malloc(Posts.size()*sizeof(uint));
	uint *idxi = (uint*)malloc(Posts.size()*sizeof(uint));
	double *ptSize = (double*)malloc(Posts.size()*sizeof(double));
	double *ntSize = (double*)malloc(Posts.size()*sizeof(double));
	auto itr = Posts.begin();
	for(uint i = 0; itr != Posts.end(); ++itr, ++i){
		// i-th post
		idxu[i] = itr->first.x;
		idxi[i] = itr->first.y;
		ptSize[i] = (itr->second).size();
		ntSize[i] = tagSize - ptSize[i];
		uint m = 0;
		uint n = 0;
		for(uint t = 0; t < tagSize; ++t){
			if( (itr->second).end() != (itr->second).find(t) ){
				idxtpos[i*tagSize + m++] = t;
			}else{
				idxtneg[i*tagSize + n++] = t;
			}
		}
	}
	//-------------------------
	// initialize U,I,T
	//-------------------------
    DMatrix MatUser(factorSize, userSize);
    DMatrix MatItem(factorSize, itemSize);
	DMatrix MatTag(factorSize, tagSize);
	char *type; 
	type = mxArrayToString(prhs[5]); 
	if(!strcmp(type,"Load")){
		MatUser.load("User.txt");
		MatItem.load("Item.txt");
		MatTag.load("Tag.txt");
	}else{
		MatUser.init(INIT_RAND_N, mean, stdev);
		MatItem.init(INIT_RAND_N, mean, stdev);
		MatTag.init(INIT_RAND_N, mean, stdev);
	}
	// SGD method to update factor matrices
	double *inter_v = (double*)malloc(factorSize*sizeof(double));
	double *inter_q = (double*)malloc(factorSize*sizeof(double));
	// w(t+,t-) = s(y(u,i,t+,t-))[1- s(y(u,i,t+,t-))]
	double *wtptn = (double*)malloc(tagSize*tagSize*sizeof(double));
	memset(wtptn, 0, tagSize*tagSize*sizeof(double));
	double *sumwtpos = (double*)malloc(tagSize*sizeof(double));
	double *sumwtneg = (double*)malloc(tagSize*sizeof(double));
	memset(sumwtpos, 0, tagSize*sizeof(double));
	memset(sumwtneg, 0, tagSize*sizeof(double));
    for(uint l = 0,saveflag = 0; l < maxiter;++l){
		//for (u,i) in Posts do
		start = clock();
		// objective function
		double Objfun = 0.0;
		Objfun -= Frobenius(MatUser);
		Objfun -= Frobenius(MatItem);
		Objfun -= Frobenius(MatTag);
		Objfun *= lambda;
		for(uint s = 0; s < Posts.size(); ++s){
			// auc = (1/z) * \sum{t+}\sum{t-}w(t+,t-)s(y(u,i,t+,t-))
			double auc = 0.0;			
			uint u = idxu[s];
			uint i = idxi[s];
			uint psize = ptSize[s];
			uint nsize = ntSize[s];
			double z = (double)( psize * nsize );
			// compute the wtptn before each iteration
			// wtptn is computed is first, so the old value will be updated
			memset(inter_v, 0, factorSize*sizeof(double));
			memset(inter_q, 0, factorSize*sizeof(double));
			memset(sumwtpos, 0, tagSize*sizeof(double));
			memset(sumwtneg, 0, tagSize*sizeof(double));
			// inter_v(f) = \sum{t+,t-} w(t+,t-)(t_{f,t+} - t_{f,t-})
			for(uint m = 0; m < psize; ++m){
				uint tag_pos = idxtpos[s*tagSize + m];
				for(uint n = 0; n < nsize; ++n){
					uint tag_neg = idxtneg[s*tagSize + n];
					double y = diff_y_pred(MatUser, MatItem, MatTag, u, i, tag_pos, tag_neg);
					double wtptn_ui = sigmoid(y)*(1 - sigmoid(y));
					wtptn[m * psize + n] = wtptn_ui;
					sumwtneg[m] += wtptn_ui;
					sumwtpos[n] += wtptn_ui;
					auc += sigmoid(y);
					for(uint f = 0; f < factorSize; ++f){
						inter_v[f] += wtptn_ui*(MatTag(f,tag_pos) - MatTag(f,tag_neg));
					}
				}
			}
			for(uint f = 0; f < factorSize; ++f){
				inter_q[f] = MatUser(f,u)*MatItem(f,i);
			}
			Objfun += auc/z;
			//---------------------------------
            // update user and item matrices
			//---------------------------------
            for(uint f = 0; f < factorSize; ++f){
                MatUser(f,u) += alpha*(MatItem(f,i)*inter_v[f]/z - lambda*MatUser(f,u));
                MatItem(f,i) += alpha*(MatUser(f,u)*inter_v[f]/z - lambda*MatItem(f,i));
            }
			//----------------------------
			// update the postive tag
			//----------------------------
			for(uint m = 0; m < psize; ++m){
				uint tag_pos = idxtpos[s*tagSize + m];
				double gradt = sumwtneg[m]/z;
				for(uint f = 0; f < factorSize; ++f){
					MatTag(f,tag_pos)  += alpha*(inter_q[f]*gradt - lambda*MatTag(f,tag_pos));
				}
			}
			//----------------------------
			// update the negative tag
			//----------------------------
			for(uint n = 0; n < nsize; ++n){
				uint tag_neg = idxtneg[s*tagSize + n];
				double gradt = sumwtpos[n]/z;
				for(uint f = 0; f < factorSize; ++f){
					MatTag(f,tag_neg)  += alpha*(-inter_q[f]*gradt - lambda * MatTag(f,tag_neg));
				}
			}
        }
		++saveflag;
		if(saveflag > 100){
			saveflag = 0;
			MatUser.save("User.txt");
			MatItem.save("Item.txt");
			MatTag.save("Tag.txt");
		}
		mexPrintf("Iter:%d,ObjectFuntcionValue:%f, TimeDuration:%f\n",l,Objfun,timeDuration(start));mexEvalString("drawnow");
    }
	MatUser.save("User.txt");
	MatItem.save("Item.txt");
	MatTag.save("Tag.txt");
	///////////////////
	// Conveter
	//////////////////
	plhs[0] = mxCreateDoubleMatrix(factorSize, userSize, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(factorSize, itemSize, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(factorSize, tagSize, mxREAL);
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
	free(idxu);
	free(idxi);
	free(idxtpos);
	free(idxtneg);
	free(ptSize);
	free(ntSize);
	free(sumwtneg);
	free(sumwtpos);
	mxFree(type);
}
