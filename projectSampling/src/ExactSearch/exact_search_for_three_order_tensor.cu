#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
// includes CUDA Runtime
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// find top 1024 value
#define top_t 1024

const dim3 blockSize(16, 16, 1);
const dim3 gridSize(64, 64, 1);
const size_t BlockSize = gridSize.x * gridSize.y;

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

/* 
	cuSubIndex is used for loop
	data: 
		x, y: indicate the current index;
		xLoop, yLoop: the max for x and y;
*/
struct cuSubIndex
{
	size_t x;
	size_t y;
	size_t xLoop;
	size_t yLoop;
	bool isDone;
	__device__ cuSubIndex(size_t a,size_t b){
		x = 0;
		y = 0;
		xLoop = a;
		yLoop = b;
		isDone = false;
	}
	__device__ ~cuSubIndex(){}
	__device__ cuSubIndex& operator++(){
    	++x;
    	if(x == xLoop){
    		x = 0;
    		++y;
    	}
    	if(y == yLoop){
    		isDone = true;
    	}
    	return (*this);
	}
};


/* 
	matrix
*/
typedef struct 
{
	// dimension of each vector
	size_t dim;
	// number of vectors
	size_t num;
	// element of the matrix
	float *element;
}Matrix;

/*
	insert the value to a sorted list
	value: the value to be inserted
	toInset: the list
	num: the length of list
*/
__device__ int doInsert(float value, float* toInsert, int num){
	float front,next;
	for(int i = 0; i < num; ++i){
		// find where to insert
		if(value > toInsert[i]){
			// insert the value before i
			front = toInsert[i];
			toInsert[i] = value;
			// shift the left element
			for(int j = i + 1; j < num; ++j){
				next = toInsert[j];
				toInsert[j] = front;
				front = next;
			}
			// return the insert position
			return i;
		}

	}
	return num;
}

/*
	h_maxValue has size gridSize*top-t;
	each block will compute their own top-t value and 
	then save into the corresponding position in h_maxValue;
	numXaxis, numYaxis : the number of vector of 
	Matrix X and Matrix Y;
*/
__global__ void GetMaxValue(Matrix *dev_data,\
							size_t numXaxis, \
							size_t numYaxis, \
							int dim, \
							float *h_maxValue);
/*
	see h_maxValue has gridSize vectors
	for each vector containing top-t values in h_maxValue
	find how many values the vector has that has the potential 
	to be in order top-t in global;
*/
__global__ void getPot(float*h_maxValue,int*potIdxBlock);

/*
	merge the h_maxValue into one vector (top-t)
	according to the potential we get;
*/
__global__ void mergeSort(float*h_maxValue,float*dst,int*potIdxBlock);

/*	
	main function
	input: data-filename, out-filename
*/
int main(int argc, char const *argv[])
{
	//----------------------------
	// check the input arguments
	//----------------------------
	if(argc <= 2){ 
		printf("Usage: exact_search <filename>\n");
		return -1;
	}
	char datafile[1024];
	char resultfile[1024];
	strcpy(datafile, argv[1]);
	strcpy(resultfile, argv[2]);
	printf("input file %s\n",datafile);
	FILE *fp;
	if((fp = fopen(datafile,"r") )== NULL){
		printf("file %s\"can not be find\"\n",datafile);
		return 1;
	}

	//------------------------------------------
	// file formate:
	// line 1: num_0 num_1 num_2
	// line 2: dim
	// line 3: 0-th vector of 0-th vector set 
	//------------------------------------------

	//-----------------------------------
	// Load the numMat and dim from file
	//-----------------------------------
	printf(">> Loading data from file %s!\n", datafile);
	int numMat = 0;
	int dim = 0;
	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	if(readline(fp) != NULL){
		char *p = strtok(line, " \t");
		++numMat;
		if(p == NULL || *p == '\n'){
			printf("Not NULL");
		}		
		while(1){
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n'){
				break;
			}
			++numMat;
		}
	}
	if(numMat != 3){
		fclose(fp);
		printf("Wrong formate or No data\n");
		return 0;
	}
	size_t *max_index = (size_t *)malloc(numMat*sizeof(size_t));
	rewind(fp);
	if(readline(fp) != NULL){
		char *p = strtok(line, " \t");
		max_index[0] = atoi(p);
		for(int i = 1; i < numMat; ++i){
			char *p = strtok(NULL," \t");
			max_index[i] = atoi(p);
		}
	}
	if(readline(fp) != NULL){
		dim = atoi(line);
	}
	//-----------------------------------
	// Load data's element from file
	//-----------------------------------
	Matrix *data = (Matrix*)malloc(numMat*sizeof(Matrix));
	for(int i = 0; i < numMat; ++i){
		data[i].num = max_index[i];
		data[i].dim = dim;
		data[i].element = (float*)malloc(max_index[i]*dim*sizeof(float));
		for(size_t j = 0; j < max_index[i]; ++j){
			char *p;
			if(readline(fp) != NULL){
				p = strtok(line," \t");
				data[i].element[j*dim] = strtod(p,NULL);
			}
			for(size_t k = 1; k < dim; ++k){
				p = strtok(NULL,"\t");
				data[i].element[j*dim + k] = strtod(p, NULL);
			}
		}
	}
	fclose(fp);
	//-----------------------------------
	// Copy the data in host to device
	//-----------------------------------
	printf(">> Copying data to device!\n");
	Matrix *h_data = (Matrix*)malloc(numMat*sizeof(Matrix));
	memcpy(h_data, data, numMat *sizeof(Matrix));
	for(int i = 0; i < numMat; ++i){
		cudaMalloc( &(h_data[i].element), \
					data[i].dim*data[i].num*sizeof(float));
		cudaMemcpy( h_data[i].element, \
					data[i].element, \
					data[i].dim*data[i].num*sizeof(float), \
					cudaMemcpyHostToDevice);
	}
	Matrix* dev_data;
	cudaMalloc(&dev_data, numMat*sizeof(Matrix));
	cudaMemcpy(dev_data, h_data, numMat*sizeof(Matrix), cudaMemcpyHostToDevice);
	
	printf(">> Loading data completed!\n");
	//--------------------------------------------------------------
	// Create variables to save the top 1024 value of each block
	//--------------------------------------------------------------
	printf(">> Starting search!\n");
	float *h_maxValue,*d_maxValue;
	float *h_top_t, *d_top_t;
	// d_maxValue is the place to save the top-t values of each block
	cudaMalloc(&d_maxValue, BlockSize*top_t*sizeof(float));
	cudaMalloc(&h_top_t, top_t*sizeof(float));
	d_top_t = (float*)malloc(top_t*sizeof(float));
	//------------------------------------------------------------
	// Invoke kernel
	//------------------------------------------------------------
	GetMaxValue<<<gridSize,blockSize>>>(dev_data, \
									    data[0].num, data[1].num, \
										dim, \
										d_maxValue);
	cudaDeviceSynchronize();
	int *potIdxBlock;
	cudaMalloc(&potIdxBlock, BlockSize*sizeof(int));
	getPot<<<gridSize,1>>>(d_maxValue,potIdxBlock);
	cudaDeviceSynchronize();
	mergeSort<<<1,1>>>(d_maxValue,d_top_t,potIdxBlock);
	//--------------------------------------------------
	// Copy result from device memory to host memory
	//--------------------------------------------------
	cudaMemcpy(h_top_t, d_top_t, top_t*sizeof(float),cudaMemcpyDeviceToHost);
	//----------------------------
	// write the top-t value to file
	//----------------------------
	FILE *fp_out = fopen(resultfile, "w");
	for (int i = 0; i < top_t; ++i){
		fprintf(fp_out,"%f\n",h_top_t[i]);
	}
	fclose(fp_out);
	//-----------------------------------
	// Free device memory and host memory
	//-----------------------------------
	for (int i = 0; i < numMat; ++i){
		cudaFree(h_data[i].element);
		free(data[i].element); 
	}
	cudaFree(dev_data);
	cudaFree(d_maxValue);
	cudaFree(d_top_t);
	cudaFree(potIdxBlock);
	free(line);
	free(max_index);
	free(data);
	free(h_data);
	free(h_top_t);
	return 0;
}


__global__ void GetMaxValue(Matrix *dev_data,\
							size_t numXaxis, \
							size_t numYaxis, \
							int dim, \
							float *d_maxValue){
	// shared memory to save top 1024 value of each thread in this block
	__shared__ float cache[16*16*1024];
	__shared__ int potIdx[16*16];
	// the thread's potions
  	// for each thread in the block
	// the index of this thread is the block is
	// x = blockIdx.x * blockDim.x + threadIdx.x;
	// y = blockIdx.y * blockDim.y + threadIdx.y;
	const int2 thread_in_block = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
          		                            blockIdx.y * blockDim.y + threadIdx.y);
	// thread position with the matrix
	int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
          		                    blockIdx.y * blockDim.y + threadIdx.y);

  	// some intermediate variables
	float temp = 0.0;
	float *mulValue = (float*)malloc(dim*sizeof(float));
	// the result will be saved at cache_pos
  	const int cache_pos = (threadIdx.x + threadIdx.y * 16) * top_t;
 	// initialize the cache with the first value
 	// 0 may be bad for matrix are not guarantee to be positive
 	for(int d = 0; d < dim; ++d){
		temp +=  dev_data[0].element[thread_2D_pos.x*dim + d] *\
		           	dev_data[1].element[thread_2D_pos.y*dim + d]*\
		           	dev_data[2].element[d];
	} 	
	for(int i = 0; i < top_t; ++i){
		cache[cache_pos + i] = temp;
	}
  	// for there are too many works, thread will
  	// do a sub work xLoop * yLoop times 
  	size_t xLoop = (numXaxis + 16*64 - 1) / 16*64;
  	size_t yLoop = (numYaxis + 16*64 - 1) / 16*64;
  	// indicate the current work progress
  	struct cuSubIndex subIndex(xLoop,yLoop);

  	// do loop
  	while(!subIndex.isDone){
  		thread_2D_pos.x = thread_in_block.x + subIndex.x * 16*64;
  		thread_2D_pos.y = thread_in_block.y + subIndex.y * 16*64;
  		// out of range
  		if(thread_2D_pos.x >= numXaxis || thread_2D_pos.y >= numYaxis){
  			++subIndex;
  			continue;
  		}
		// compute X(i, j, k) for k in the range of Set C
		// i = thread_2D_pos.x
		// j = thread_2D_pos.y
		for(int d = 0; d < dim; ++d){
			mulValue[d] =  dev_data[0].element[thread_2D_pos.x*dim + d] *\
			           	dev_data[1].element[thread_2D_pos.y*dim + d];
		}
		// compute X(i, j, :) and put into the cache
		for(size_t k = 0; k < dev_data[2].num; ++k){
			temp = 0.0;
			for(int d = 0; d < dim; ++d){
				temp += dev_data[2].element[k*dim + d] * \
						mulValue[d];
			}
			if(temp > cache[cache_pos + top_t - 1]){
				doInsert(temp,(cache+cache_pos),top_t);
			}
		}
		++subIndex;
	}
	__syncthreads();

	//------------------------------------------------------
	// merge the cache to one vector containing the top-t value
	//------------------------------------------------------

	// there are 16 * 16 vectors,
	// for each vector called a beam
	// 16 * 16 is the same size of thread size per block;
	// each thread process on beam
	// find the top-s value in a beam are potential to 
	// be top-t of global
	// using all of top-s values in every beam 
	// we can get the global top-t values of this block
	int index = 0;
	bool doneSearch = false;
	for(int i = 0; i < top_t; ++i){
		// the order of cache[cache_pos + i] is at least i
		temp = cache[cache_pos + i];
		index = i;
		doneSearch = false;
		// compare this value to others
		// if encounter a value is other beam which bigger than it
		// increment index and go on, otherwise 
		// encountering a value no bigger than it
		// go to another beam to do comparison
		for(int m = 0; m < 16; ++m){
			for(int n = 0; n < 16; ++n){
				// escape self beam
				if(m == threadIdx.x && n == threadIdx.y){
					continue; 
				}
				// search in other beam
				for(int k = 0; k < top_t; ++k){
					// encounter a bigger value 
					if(temp < cache[(m + n * 16) * 1024 + k]){
						++index;
					}else{
						// this value is bigger than the left
						// of this beam, so no need to search deeper
						break;
					}
					// if this value is out of the top-t
					// we can finish searching
					if(index >= top_t){
						doneSearch = true;
						break;
					}
				}
				if(doneSearch) break;
			}
			if(doneSearch){
				break;
			}
		}
		// if beam[i] has no potential to be the top-t
		// then the number of elements have the potential
		// is i then record it;
		if(doneSearch){
			potIdx[threadIdx.x + threadIdx.y * 16] = i;
			break;
		}
  	}
  	
  	__syncthreads();
  	if(threadIdx.x == 0 && threadIdx.y == 0){
  		int count = 0;
 		for(int m = 0; m < 16; ++m){
			for(int n = 0; n < 16; ++n){
				count += potIdx[threadIdx.x + threadIdx.y * 16];
			}
		}
		float *potV = (float*)malloc(count*sizeof(float));
		int p = 0;
 		for(int m = 0; m < 16; ++m){
			for(int n = 0; n < 16; ++n){
				for(int i = 0 ; i < potIdx[threadIdx.x + threadIdx.y * 16];++i){
					potV[p] = cache[(m + n * 16) * 1024 + i];
					++p;
				}
			}
		}
		//sort potV
		for(int i = 1; i < count; i++){  
			if(potV[i] > potV[i-1]){
			    int j = i - 1;
			    float x = potV[i];
			    potV[i] = potV[i-1];
			    while(j >=0 && x > potV[j]){
			        potV[j + 1] = potV[j];
			        j--;
			    }
			    potV[j+1] = x;
			}  
		}
		for(int i = 0; i < top_t; ++i){
			d_maxValue[(blockIdx.x + blockIdx.y * 64)*1024 + i] = potV[i];
		}
		free(potV);
  	}
	free(mulValue);
}

__global__ void getPot(float*src, int *potIdxBlock){
	double temp;
	int index = 0;
	bool doneSearch = false;
	size_t pos = (blockIdx.x + blockIdx.y*64)*top_t;
	float *potV = (float*)malloc(64*64*sizeof(float));
	for(int i = 0; i < top_t; ++i){
		// the order of cache[cache_pos + i] is at least i
		temp = src[pos + i];
		index = i;
		doneSearch = false;
		// compare this value to others
		// if encounter a value is other beam which bigger than it
		// increment index and go on, otherwise 
		// encountering a value no bigger than it
		// go to another beam to do comparison
		for(int m = 0; m < blockDim.x; ++m){
			for(int n = 0; n < blockDim.y; ++n){
				// escape self beam
				if(m == blockDim.x && n == blockDim.y){
					continue; 
				}
				// search in other beam
				for(int k = 0; k < top_t; ++k){
					// encounter a bigger value 
					if(temp < src[pos + k]){
						++index;
					}else{
						// this value is bigger than the left
						// of this beam, so no need to search deeper
						break;
					}
					// if this value is out of the top-t
					// we can finish searching
					if(index >= top_t){
						doneSearch = true;
						break;
					}
				}
				if(doneSearch) break;
			}
			if(doneSearch){
				break;
			}
		}
		// if beam[i] has no potential to be the top-t
		// then the number of elements have the potential
		// is i then record it;
		if(doneSearch){
			potIdxBlock[(blockIdx.x + blockIdx.y*64)] = i;
			break;
		}
  	}
  	
}

__global__ void mergeSort(float*src,float*dst, int *potIdxBlock){
  	int count = 0;
 	for(int m = 0; m < 64; ++m){
		for(int n = 0; n < 64; ++n){
			count += potIdxBlock[m + n * 64];
		}
	}
	float *potV = (float*)malloc(count*sizeof(float));
	int p = 0;
 	for(int m = 0; m < 64; ++m){
		for(int n = 0; n < 64; ++n){
			for(int i = 0 ; i < potIdxBlock[m + n * 64];++i){
				potV[p] = src[(m + n * 64) * 1024 + i];
				++p;
			}
		}
	}
	//sort potV
	for(int i = 1; i < count; i++){  
		if(potV[i] > potV[i-1]){
		    int j = i - 1;
		    float x = potV[i];
		    potV[i] = potV[i-1];
		    while(j >= 0 && x > potV[j]){
		        potV[j + 1] = potV[j];
		        j--;
		    }
		    potV[j+1] = x;
		}  
	}
	for(int i = 0; i < top_t; ++i){
		dst[i] = potV[i];
	}
	free(potV);
}