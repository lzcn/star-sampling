#include <map>

#include "matrix.h"
#include "mex.h"
/*  
	[indexes, rating] = merge(user, item, rating, user, item, tagID)
	merge into one tuple (indexes) matrix and rating array
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
	//------------------------
	// Initialization
	//------------------------
	size_t Num = mxGetM(prhs[0]);
	// user - movie -rating
	size_t *user = (size_t*)malloc(Num*sizeof(size_t));
	size_t *item = (size_t*)malloc(Num*sizeof(size_t));
	double *rating = mxGetPr(prhs[2]);
	for (size_t i = 0; i < Num; ++i){
		user[i] = (size_t)mxGetPr(prhs[0])[i];
		item[i] = (size_t)mxGetPr(prhs[1])[i];
	}
	// map to save <user-movie, rating>
	std::map<point2D, double> UserItemRating;
	// count the user ID
	std::map<size_t, int> UserCount;
	// count the movie ID
	std::map<size_t, int> ItemCount;
	// initialize the map of <user-movie, rating>
	for(auto i = 0; i < Num; ++i){
		UserItemRating[point2D(user[i],item[i])] = rating[i];
		UserCount[user[i]] += 1;
		ItemCount[item[i]] += 1;
	}
	// delete some <user, movie, tag> so that each user movie and tag occurred at least 5 times.
	for(;;){
		std::map<point2D,double> temp;
		// copy the UserItemRating to a temp map
		temp = UserItemRating;
		printf("The length of tuple:%d\n",temp.size());
		bool changed = false;
		// looping through each of the elements
		for(auto itr = temp.begin(); itr != temp.end(); ++itr){
			if(UserCount[itr->first.x] < 5 || ItemCount[itr->first.y] < 5){
				// delete the tuple
				UserItemRating.erase(itr->first);
				// decrement count for <user, movie, tag>
				--UserCount[itr->first.x];
				--ItemCount[itr->first.y];
				// switch the state to changed
				changed = true;
			}
		}
		// if element changed in the loop, then complete
		if(changed) continue;
		else break;
	}

	// range compression
	std::map<size_t,size_t> UserMap;
	std::map<size_t,size_t> ItemMap;
	std::map<size_t,size_t> TagMap;
	// find the ID has occurred
	for(auto itr = UserItemRating.begin(); itr != UserItemRating.end(); ++itr){
		UserMap[itr->first.x] = 1;
		ItemMap[itr->first.y] = 1;
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
	// do compression
	size_t num = UserItemRating.size();
	plhs[0] = mxCreateNumericMatrix(num, 2, mxUINT64_CLASS, mxREAL);
	uint64_T* plhs_indexes = (uint64_T*)mxGetData(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(num, 1, mxREAL);
	double *plhs_rating = mxGetPr(plhs[1]);
	offset = 0;
	for(auto itr = UserItemRating.begin(); itr != UserItemRating.end(); ++itr){
		plhs_indexes[offset] = UserMap[itr->first.x] + 1;
		plhs_indexes[num + offset] = ItemMap[itr->first.y] + 1;
		plhs_rating[offset] = itr->second;
		++offset;
	}
	//--------------
	// free
	//--------------
	free(user);
	free(item);
}
