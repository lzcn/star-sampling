#include <map>

#include "matrix.h"
#include "mex.h"
/*  
	[indexes, rating] = merge(userIDa, movieIDa, rating, userIDb, movieIDb, tagID)
	merge into one tuple (indexes) matrix and rating array
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
	//------------------------
	// Initialization
	//------------------------
	size_t numRating = mxGetM(prhs[0]);
	size_t numTag = mxGetM(prhs[3]);
	// user - movie -rating
	size_t *userIDa = (size_t*)malloc(numRating*sizeof(size_t));
	size_t *movieIDa = (size_t*)malloc(numRating*sizeof(size_t));
	double *rating = mxGetPr(prhs[2]);
	for (size_t i = 0; i < numRating; ++i){
		userIDa[i] = (size_t)mxGetPr(prhs[0])[i];
		movieIDa[i] = (size_t)mxGetPr(prhs[1])[i];
	}
	// user - movie - tag
	size_t *userIDb = (size_t*)malloc(numTag*sizeof(size_t));
	size_t *movieIDb = (size_t*)malloc(numTag*sizeof(size_t));
	size_t *tagID = (size_t*)malloc(numTag*sizeof(size_t));

	for (size_t i = 0; i < numTag; ++i){
		userIDb[i] = (size_t)mxGetPr(prhs[3])[i];
		movieIDb[i] = (size_t)mxGetPr(prhs[4])[i];
		tagID[i] = (size_t)mxGetPr(prhs[5])[i];
	}
	// map to save <user-movie-tag, rating>
	std::map<point3D, double> UserMovieTag;
	// map to save <user-movie, rating>
	std::map<point2D, double> UserMovieRating;
	// count the user ID
	std::map<size_t, int> UserCount;
	// count the movie ID
	std::map<size_t, int> MovieCount;
	// count the tag ID
	std::map<size_t, int> TagCount;
	// initialize the map of <user-movie, rating>
	for(auto i = 0; i < numRating; ++i){
		UserMovieRating[point2D(userIDa[i],movieIDa[i])] = rating[i];
	}
	// initialize the map of <user-movie-tag, rating>
	for(auto i = 0; i < numTag; ++i){
		// only when <user-movie> has occurred in map UserMovieRating
		auto itr = UserMovieRating.find(point2D(userIDb[i],movieIDb[i]));
		if(itr != UserMovieRating.end()){
			UserMovieTag[point3D(userIDb[i],movieIDb[i],tagID[i])] = itr->second;\
			// count user, movie, tag
			UserCount[userIDb[i]] += 1;
			MovieCount[movieIDb[i]] += 1;
			TagCount[tagID[i]] += 1;
		}
	}
	// delete some <user, movie, tag> so that each user movie and tag occurred at least 5 times.
	for(;;){
		std::map<point3D,double> temp;
		// copy the UserMovieTag to a temp map
		temp = UserMovieTag;
		bool changed = false;
		// looping through each of the elements
		for(auto itr = temp.begin(); itr != temp.end(); ++itr){
			if(UserCount[itr->first.x] < 5 || MovieCount[itr->first.y] < 5 || TagCount[itr->first.z] < 5){
				// delete the tuple
				UserMovieTag.erase(itr->first);
				// decrement count for <user, movie, tag>
				--UserCount[itr->first.x];
				--MovieCount[itr->first.y];
				--TagCount[itr->first.z];
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
	std::map<size_t,size_t> MovieMap;
	std::map<size_t,size_t> TagMap;
	// find the ID has occurred
	for(auto itr = UserMovieTag.begin(); itr != UserMovieTag.end(); ++itr){
		UserMap[itr->first.x] = 1;
		MovieMap[itr->first.y] = 1;
		TagMap[itr->first.z] = 1;
	}
	// record the compression map
	size_t offset = 0;
	for(auto itr = UserMap.begin(); itr != UserMap.end(); ++itr){
		itr->second = offset++;
	}
	offset = 0;
	for(auto itr = MovieMap.begin(); itr != MovieMap.end(); ++itr){
		itr->second = offset++;
	}
	offset = 0;
	for(auto itr = TagMap.begin(); itr != TagMap.end(); ++itr){
		itr->second = offset++;
	}
	// do compression
	size_t num = UserMovieTag.size();
	plhs[0] = mxCreateNumericMatrix(num, 3, mxUINT64_CLASS, mxREAL);
	uint64_T* plhs_indexes = (uint64_T*)mxGetData(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(num, 1, mxREAL);
	double *plhs_rating = mxGetPr(plhs[1]);
	offset = 0;
	for(auto itr = UserMovieTag.begin(); itr != UserMovieTag.end(); ++itr){
		plhs_indexes[offset] = UserMap[itr->first.x] + 1;
		plhs_indexes[num + offset] = MovieMap[itr->first.y] + 1;
		plhs_indexes[num + num + offset] = TagMap[itr->first.z] + 1;
		plhs_rating[offset++] = itr->second;
	}
	//--------------
	// free
	//--------------
	free(userIDa);
	free(userIDb);
	free(movieIDa);
	free(movieIDb);
	free(tagID);
}