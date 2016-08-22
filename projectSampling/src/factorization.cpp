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
	return sqrt(ans);
}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
    //--------------------------------------------
    // Remove some tuple untill to be 5-core data
    //--------------------------------------------
	clock_t start;
	uint NumofEntry = mxGetM(prhs[0]);
	uint *userid = (uint*)malloc(NumofEntry*sizeof(uint));
	uint *itemid = (uint*)malloc(NumofEntry*sizeof(uint));
    uint *tagid  = (uint*)malloc(NumofEntry*sizeof(uint));
	for (uint i = 0; i < NumofEntry; ++i){
		userid[i] = (uint)mxGetPr(prhs[0])[i];
		itemid[i] = (uint)mxGetPr(prhs[1])[i];
		tagid[i]  = (uint)mxGetPr(prhs[2])[i];
	}
    // record the data to a map, and count user,item,tag
	std::set<point3D> UserItemTag;
	std::map<uint, int> UserCount;
	std::map<uint, int> ItemCount;
	std::map<uint, int> TagCount;
	for(auto i = 0; i < NumofEntry; ++i){
        UserItemTag.insert( point3D(userid[i],itemid[i],tagid[i]) );
        UserCount[userid[i]] += 1;
        ItemCount[itemid[i]] += 1;
        TagCount[ tagid[i] ] += 1;
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
	typedef std::vector<uint> posTag;
    std::map<point2D, posTag> Posts;
	
	for(auto itr = UserItemTag.begin(); itr != UserItemTag.end(); ++itr){
        uint user = UserMap[itr->x];
        uint item = ItemMap[itr->y];
        uint tag  = TagMap[itr->z];
        Posts[point2D(user,item)].push_back(tag);
	}
	mexPrintf("Posts size :%d\n",Posts.size());mexEvalString("drawnow");
	return;
    //////////////////////////////
    // SGD for factorization
    //////////////////////////////
    // initialized the factor matrices
    uint userSize = UserMap.size();
    uint itemSize = ItemMap.size();
    uint tagSize = TagMap.size();
    uint factorSize = 32;
    double mean = 0.25;
    double stdev = 1e-2;
    double lambda = 1e-6;
    double alpha = 0.1;
    uint maxiter = 1;
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
    for(uint l = 0; l < maxiter;++l){
		//for (u,i) in Posts do
		start = clock();
		std::map<point2D,double> wtptn;
		uint u,i;
		double AUC = 0.0;
        for(auto itr = Posts.begin();itr != Posts.end(); ++itr,++offset){
			double AUCui = 0.0;
            u = itr->first.x;
            i = itr->first.y;
			posTag ptagVec = itr->second;
			auto tag_pos = ptagVec.begin();
			// compute the intermediate vector v
			memset(inter_v, 0, factorSize*sizeof(double));
			for (tag_pos = ptagVec.begin(); tag_pos != ptagVec.end(); ++tag_pos) {
				for (uint tag_neg = 0; tag_neg < tagSize; ++tag_neg){
					if(ptagVec.end() != std::find(ptagVec.begin(),ptagVec.end(),tag_neg)){ continue; }
					double y = diff_y_pred(MatUser,MatItem,MatTag,u,i,*tag_pos,tag_neg);
					double temp = sigmoid(y)*(1-sigmoid(y));
					AUCui += sigmoid(y);
					wtptn[point2D(*tag_pos,tag_neg)] = temp;
					for(uint f = 0; f < factorSize; ++f){
						inter_v[f] += temp*(MatTag(f,*tag_pos)-MatTag(f,tag_neg));
					}
				}
			}
			// compute the intermediate vector q
			for(uint f = 0; f < factorSize; ++f){
				inter_q[f] = MatUser(f,u)*MatItem(f,i);
			}
			double gradu = 0.0, gradi = 0.0;
			double z = 1.0 / (double)( ptagVec.size() * (tagSize-ptagVec.size()) );
			for(uint f = 0; f < factorSize; ++f){
				gradu += MatItem(f,i)*inter_v[f];
				gradi += MatUser(f,u)*inter_v[f];
			}
			gradu *= z;
			gradi *= z;
			AUCui *= z;
            // update user and item matrices
            for(uint r = 0; r < factorSize; ++r){
                MatUser(r,u) += alpha*(gradu - lambda*MatUser(r,u));
                MatItem(r,i) += alpha*(gradi - lambda*MatItem(r,i));
            }
			// update tag matrix
            for(uint t = 0; t < tagSize; ++t){
				double gradt = 0.0;
				if(ptagVec.end() != std::find(ptagVec.begin(),ptagVec.end(),t)){
					for(uint tag_neg = 0; tag_neg < tagSize; ++tag_neg){
						if(ptagVec.end() != std::find(ptagVec.begin(),ptagVec.end(),tag_neg)){ continue; }
						gradt += wtptn[point2D(t,tag_neg)];
					}
					gradt = -1.0 * z * gradt;
				}else{// if t is negative tag loop in positive tag
					for(tag_pos = ptagVec.begin(); tag_pos != ptagVec.end(); ++tag_pos){
						gradt += wtptn[point2D(*tag_pos,t)];
					}
					gradt = -1.0 * z * gradt;
				}
                for(uint f = 0; f < factorSize; ++f){
					MatTag(f,t)  += alpha*(inter_q[f]*gradt - lambda*MatTag(f,t));
                }
            }
			AUC += AUCui;
        }
		double regularTerm = 0.0;
		regularTerm += Frobenius(MatUser);
		regularTerm += Frobenius(MatItem);
		regularTerm += Frobenius(MatTag);
		regularTerm *= lambda;
		mexPrintf("Iter:%d, ObjectFuntcionValue:%f, TimeDuration:%f\n",l,AUC-regularTerm,timeDuration(start));mexEvalString("drawnow");
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
	free(userid);
	free(itemid);
	free(tagid);
	free(inter_q);
	free(inter_v);
}
