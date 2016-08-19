#include <map>
#include <cstdio>
#include <ctime>
#include "dmatrix.h"
#include "util.h"
#include "coordinate.h"
#include "random.h"
#include "mex.h"
double diff_y_pred(DMatrix &U,DMatrix &I,DMatrix &T,size_t u,size_t i,size_t pos,size_t neg){
	size_t rank = U.row;
	double ans = 0.0;
	for(size_t r = 0; r < rank; ++r){
		ans += U(r,u)*I(r,i)*(T(r,pos)-T(r,neg));
	}
	return ans;
}
double Frobenius(DMatrix &A){
	double ans = 0.0;
	for(size_t i = 0; i < A.col*A.row; i++){
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
	size_t NumofEntry = mxGetM(prhs[0]);
	size_t *userid = (size_t*)malloc(NumofEntry*sizeof(size_t));
	size_t *itemid = (size_t*)malloc(NumofEntry*sizeof(size_t));
    size_t *tagid  = (size_t*)malloc(NumofEntry*sizeof(size_t));
	for (size_t i = 0; i < NumofEntry; ++i){
		userid[i] = (size_t)mxGetPr(prhs[0])[i];
		itemid[i] = (size_t)mxGetPr(prhs[1])[i];
		tagid[i]  = (size_t)mxGetPr(prhs[2])[i];
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
		bool changed = false;
		mexPrintf("Current length of tuple:%d\n", temp.size());
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
    // map to save <posts, posTag>
	typedef std::vector<size_t> posTag;
    std::map<point2D, posTag> Posts;
	for(auto itr = UserItemTag.begin(); itr != UserItemTag.end(); ++itr){
        size_t user = UserMap[itr->first.x];
        size_t item = ItemMap[itr->first.y];
        size_t tag  = TagMap[itr->first.z];
        Posts[point2D(user,item)].push_back(tag);
	}
	mexPrintf("Posts size :%d\n",Posts.size());

    //////////////////////////////
    // SGD for factorization
    //////////////////////////////
    // initialized the factor matrices
    size_t userSize = UserMap.size();
    size_t itemSize = ItemMap.size();
    size_t tagSize = TagMap.size();
    size_t factorSize = 64;
    double mean = 0.25;
    double stdev = 1e-2;
    double lambda = 1e-6;
    double alpha = 0.1;
    size_t maxiter = 2;
	plhs[0] = mxCreateDoubleMatrix(factorSize, userSize, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(factorSize, itemSize, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(factorSize, tagSize, mxREAL);
    DMatrix MatUser( factorSize, userSize);
    MatUser.init(INIT_RAND_N,mean,stdev);
    DMatrix MatItem(factorSize, itemSize);
    MatItem.init(INIT_RAND_N, mean, stdev);
    DMatrix MatTag(factorSize, tagSize);
    MatTag.init(INIT_RAND_N, mean, stdev);
	// SGD method to update factor matrices
    for(size_t l = 0; l < maxiter;++l){
		double AUC = 0.0;
		//for (u,i) in Posts do
        for(auto itr = Posts.begin();itr != Posts.end(); ++itr){
			start = clock();
            size_t u = itr->first.x;
            size_t i = itr->first.y;
			posTag ptagVec = itr->second;
            double z = 1.0 / (double)( ptagVec.size() * (tagSize-ptagVec.size()) );
            // update user and item matrices
            for(size_t r = 0; r < factorSize; ++r){
                double gradu = 0.0, gradi = 0.0;
                // compute the gradient
				// gradu = z * \sum_{t+} \sum_{t-} \sum_{r} 
				// w_{t+,t-} * U(u,r) * I(i,r) * (T(t+,r)-T(t-,r))
                for(auto itrtag = ptagVec.begin(); itrtag != ptagVec.begin();++itrtag){
					// loop in tag positive
                    size_t tag_pos = *itrtag;
                    for(size_t tag_neg = 0; tag_neg < tagSize; ++tag_neg){
                        if(ptagVec.end() != std::find(ptagVec.begin(),ptagVec.end(),tag_neg))
                            continue;
						// loop in tag negative
						// compute w_{t+,t-} = s(y_diff)(1- s(y_diff))
                        double y = diff_y_pred(MatUser,MatItem,MatTag,u,i,tag_pos,tag_neg);
						double logfun = sigmoid(y);
                        double wtptn = logfun*(1-logfun);
						AUC += logfun;
						double *user = &MatUser.value[u*factorSize];
						double *item = &MatItem.value[i*factorSize]; 
						double *tag1 = &MatTag.value[tag_pos*factorSize];
						double *tag2 = &MatTag.value[tag_neg*factorSize];
                        for(size_t f = 0;f < factorSize; ++f){
							double diff_tag = (*(tag1+f)-*(tag2+f));
							gradu += (*(item+f)) * diff_tag;
							gradi += (*(user+f)) * diff_tag;
                            //gradu += wtptn*MatItem(i,f)*(MatTag(tag_pos,f)-MatTag(tag_neg,f));
                            //gradi += wtptn*MatUser(u,f)*(MatTag(tag_pos,f)-MatTag(tag_neg,f));
                        }
						gradu *= wtptn;
						gradi *= wtptn;
                    }
                }
				// use grad to update
                MatUser(r,u) += alpha*(z*gradu - lambda*MatUser(r,u));
                MatItem(r,i) += alpha*(z*gradi - lambda*MatItem(r,i));

            }
            // update tag matrix
            for(size_t t = 0; t < tagSize; ++t){
                for(size_t r = 0; r < factorSize; ++r){
                    double gradt = 0.0;
					// if positive tag
                    if(ptagVec.end() != std::find(ptagVec.begin(),ptagVec.end(),t)){
						// loop in negative tag
                        for(size_t tag_neg = 0; tag_neg < tagSize; ++tag_neg){
                            if(ptagVec.end() != std::find(ptagVec.begin(),ptagVec.end(),tag_neg))
								continue;
                            double y = diff_y_pred(MatUser,MatItem,MatTag,u,i,t,tag_neg);
                            double wtptn = sigmoid(y)*(1.0-sigmoid(y));
							double *user = &MatUser.value[u*factorSize];
							double *item = &MatItem.value[i*factorSize]; 
                            for(size_t f = 0;f < factorSize; ++f){
								gradt += (*(item+f)) * (*(user+f));
                                //gradt += MatUser(u,f)*MatItem(i,f);
                            }
							gradt *= wtptn;
                        }
						gradt = -1.0 * gradt;
                    }else{
						// if negative tag loop in positive tag
                        for(auto itrtag_pos = ptagVec.begin(); itrtag_pos != ptagVec.begin();++itrtag_pos){
							double y = diff_y_pred(MatUser,MatItem,MatTag,u,i,*itrtag_pos,t);
                            double wtptn = sigmoid(y)*(1.0-sigmoid(y));
							//double *user = &MatUser.value[u*factorSize];
							//double *item = &MatItem.value[i*factorSize]; 
                            for(size_t f = 0;f < factorSize; ++f){
								//gradt += (*(item+f)) * (*(user+f));
                                gradt += MatUser(f,u)*MatItem(f,i);
                            }
							gradt *= wtptn;
                        }
                    }
                    MatTag(r,t)  += alpha*(z*gradt - lambda*MatTag(r,t));
                }
            }
			mexPrintf("timeDuration: %f",timeDuration(start));
			return;
        }
		if(0 == (l / 20)){
			double regularTerm = 0.0;
			regularTerm += Frobenius(MatUser);
			regularTerm += Frobenius(MatItem);
			regularTerm += Frobenius(MatTag);
			regularTerm *= lambda;
			mexPrintf("Iter:%d,ObjectFuntcionValue:%f\n",l,AUC-regularTerm);
		}
    }
	///////////////////
	// Conveter
	//////////////////
	for(size_t i = 0; i < userSize*factorSize; ++i){
		mxGetPr(plhs[0])[i] = MatUser.value[i];
	}
	for(size_t i = 0; i < itemSize*factorSize; ++i){
		mxGetPr(plhs[1])[i] = MatItem.value[i];
	}
	for(size_t i = 0; i < userSize*factorSize; ++i){
		mxGetPr(plhs[2])[i] = MatTag.value[i];
	}
	//--------------
	// free
	//--------------
	free(userid);
	free(itemid);
	free(tagid);
}
