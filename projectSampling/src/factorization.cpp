#include <map>

#include "dmatrix.h"
#include "util.h"
#include "coordinate.h"
#include "random.h"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
    //--------------------------------------------
    // Remove some tuple untill to be 5-core data
    //--------------------------------------------
	size_t NumofEntry = mxGetM(prhs[0]);
	size_t *userid = (size_t*)malloc(NumofEntry*sizeof(size_t));
	size_t *itemid = (size_t*)malloc(NumofEntry*sizeof(size_t));
    size_t *tagid  = (size_t*)malloc(NumofEntry*sizeof(size_t));
	for (size_t i = 0; i < NumofEntry; ++i){
		userid[i] = (size_t)mxGetPr(prhs[0])[i];
		itemid[i] = (size_t)mxGetPr(prhs[1])[i];
		tagid[i]  = (size_t)mxGetPr(prhs[1])[i];
	}
    // record the data to a map, and count user,item,tag
	std::map<point3D, bool> UserItemTag;
	std::map<size_t, int> UserCount;
	std::map<size_t, int> ItemCount;
	std::map<size_t, int> TagCount;
	for(auto i = 0; i < NumofEntry; ++i){
        UserItemTag[point3D(userid[i],itemid[i],tagid[i])] = true;
        UserCount[userid[i]] += 1;
        ItemCount[itemid[i]] += 1;
        TagCount[ tagid[i] ] += 1;
	}
	// do delete
	for(;;){
		std::map<point3D,bool> temp;
        temp = UserItemTag;
		mexPrintf("Current length of tuple:%d\n", temp.size());
		bool changed = false;
		for(auto itr = temp.begin(); itr != temp.end(); ++itr){
			if(UserCount[itr->first.x] < 5 || ItemCount[itr->first.y] < 5 || TagCount[itr->first.z] < 5){
				// delete the tuple
				UserItemTag.erase(itr->first);
				// decrement count for <user, movie, tag>
				--UserCount[itr->first.x];
				--ItemCount[itr->first.y];
				--TagCount[itr->first.z];
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
	std::map<size_t,size_t> UserMap;
	std::map<size_t,size_t> ItemMap;
	std::map<size_t,size_t> TagMap;
	// find the ID has occurred
	for(auto itr = UserItemTag.begin(); itr != UserItemTag.end(); ++itr){
		UserMap[itr->first.x] = 1;
		ItemMap[itr->first.y] = 1;
		TagMap[itr->first.z] = 1;
	}
	// record the compression map
	size_t offset = 0;
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
    // map to save <posts, postTag>
    typedef std::map<size_t,bool> posTag;
    std::map<point2D, posTag> Posts;
	for(auto itr = UserItemTag.begin(); itr != UserItemTag.end(); ++itr){
        size_t user = UserMap[itr->first.x];
        size_t item = ItemMap[itr->first.y];
        size_t tag  = TagMap[itr->first.z];
        Posts[point2D(user,item)][tag] = true;
	}
    //////////////////////////////
    // SGD for factorization
    //////////////////////////////
    // initialized the factor matrices
    size_t userSize = UserMap.size();
    size_t itemSize = ItemMap.size();
    size_t tagSize = TagMap.size();
    size_t factorSize = 64;
    double mean = 0.25;
    double stdev = 1e-4;
    double lambda = 1e-6;
    // learning rate
    double alpha = 0.1;
    size_t maxiter = 1000;
    DMatrix MatUser(userSize,factorSize);
    MatUser.init(INIT_RAND_N,mean,stdev);
    DMatrix MatItem(itemSize,factorSize);
    MatItem.init(INIT_RAND_N,mean,stdev);
    DMatrix MatTag(tagSize,factorSize);
    MatTag.init(INIT_RAND_N,mean,stdev);
    for(size_t l = 0; l < maxiter;++l){
        for(auto itr = Posts.begin();itr != Posts.end(); ++itr){
            // update factor matrices
            std::map<size_t,bool> ptag = itr->second;
            size_t u = itr->first.x;
            size_t i = itr->first.y;
            double z = 1/(double)(ptag.size()*(tagSize-ptag.size()));
            // update user and item matrices
            for(size_t r = 0; r < factorSize; ++r){
                double gradu = 0.0, gradi = 0.0;
                // compute the gradient
                for(auto itrtag = ptag.begin(); itrtag != ptag.begin();++itrtag){
                    size_t tpos = itrtag->first;
                    for(size_t tneg = 0; tneg < tagSize; ++tneg){
                        if(ptag.find(tneg))
                            continue;
                        double diff_y_pred = y_pred(u,i,tpos) - y_pred(u,i,tneg);
                        double wtptn = sigmoid(diff_y_pred)(1-sigmoid(diff_y_pred));
                        for(size_t f = 0;f < factorSize; ++f){
                            gradu += wtptn*MatItem(i,f)*(MatTag(tpos,f)-MatTag(tneg,f));
                            gradi += wtptn*MatUser(u,f)*(MatTag(tpos,f)-MatTag(tneg,f));
                        }
                    }
                }
                MatUser(u,r) += alpha*(z*gradu - lambda*MatUser(u,r));
                MatItem(i,r) += alpha*(z*gradi - lambda*MatItem(i,r));
            }
            // update tag matrix
            for(size_t t = 0; t < tagSize; ++t){
                for(size_t r = 0; r < factorSize; ++r){
                    double gradt = 0.0;
                    if(ptag.find(t)){
                        for(size_t tneg = 0; tneg < tagSize; ++tneg){
                            if(ptag.find(tneg))
                            continue;
                            double diff_y_pred = y_pred(u,i,t) - y_pred(u,i,tneg);
                            double wtptn = sigmoid(diff_y_pred)(1-sigmoid(diff_y_pred));
                            for(size_t f = 0;f < factorSize; ++f){
                                gradt += wtptn*MatUser(u,f)*MatItem(tpos,f);
                            }
                        }
                    }else{
                        for(auto itrtag_pos = ptag.begin(); itrtag_pos != ptag.begin();++itrtag_pos){
                            double diff_y_pred = y_pred(u,i,itrtag_pos->second) - y_pred(u,i,t);
                            double wtptn = sigmoid(diff_y_pred)(1-sigmoid(diff_y_pred));
                            for(size_t f = 0;f < factorSize; ++f){
                                gradt += wtptn*MatUser(u,f)*MatItem(i,f);
                            }
                        }
                    }
                    MatTag(t,r)  += alpha*(z*gradt - lambda*MatTag(t,r));
                }
            }
        }
    }
	//--------------
	// free
	//--------------
	free(userid);
	free(itemid);
	free(tagid);
}
