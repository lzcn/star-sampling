#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
// includes CUDA Runtime
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// find top 1K value
#define TOP_T 1024
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

const int threadsPerBlock = 4;
const int BlockSize = 500000;

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
//------------------------------------------------
// cuSubIndex is used for loop
/* 
	data: 
		index_cur: indicate the current index for 
		index_max: each element is the number vector
				of correspongding maxtirx
	method: 
		+ :add t to current index
*/
//------------------------------------------------
struct cuSubIndex
{
	int numMat;
	size_t *index_cur;
	const size_t *index_max;
	bool isDone;
	__device__ cuSubIndex(int n, size_t *max) :numMat(n){
		index_cur = (size_t*)malloc((n + 1)*sizeof(size_t));
		//index_max = (size_t*)malloc(n*sizeof(size_t));
		index_max = max;
		for (int i = 0; i < n + 1; ++i){
			//index_max[i] = max[i];
			index_cur[i] = 0;
		}
		//index_cur[numMat] = 0;
		isDone = false;
	}
	__device__ ~cuSubIndex(){
		//free(index_max);
		free(index_cur);
	}
	__device__ cuSubIndex&  operator+(const size_t step){
		size_t a, b;
		a = step;
		b = 0;
		size_t *tmp = (size_t *)malloc(numMat*sizeof(size_t));
		for (size_t i = 0; i < numMat; ++i){
			b = a % index_max[i];
			a = a / index_max[i];
			tmp[i] = b;
			index_cur[i] += b;
			while (index_cur[i] >= index_max[i]){
				index_cur[i] -= index_max[i];
				++index_cur[i + 1];
			}
		}
		if (a > 0){
			index_cur[numMat] += a;
		}
		if (index_cur[numMat] > 0){
			isDone = true;
		}
		free(tmp);
		return (*this);
	}
};


// data struct : matrix
typedef struct 
{
	size_t dim;
	size_t num;
	float *element;
}Matrix;

__host__ __device__ int doInsert(float value, float* toInsert, int num){
	float front,next;
	for(int i = 0; i < num; ++i){
		if(value > toInsert[i]){
			// find ans insert
			front = toInsert[i];
			toInsert[i] = value;
			// shift the left element
			for(int j = i + 1; j < num; ++j){
				next = toInsert[j];
				toInsert[j] = front;
				front = next;
			}
			return i;
		}

	}
	return num;
}
void mergeSort(float*src,float*dst, int *potIdxBlock);

__global__ void GetMaxValue(Matrix *dev_data, \
						int numMat, \
						size_t *dev_max_value, \
						int d, \
						float*max_value);

__global__ void getPot(float *src, int *potIdxBlock);
int main(int argc, char const *argv[])
{
	//----------------------------
	// check the input arguements
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
	// line 1: num0	num1 ...
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
		//printf("%s",*p);
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
	if(numMat == 0){
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

	Matrix *h_data = (Matrix*)malloc(numMat*sizeof(Matrix));
	memcpy(h_data, data, numMat *sizeof(Matrix));
	// max index stored in device
	size_t *dev_index_max;
	cudaMalloc(&dev_index_max,numMat*sizeof(size_t));
	cudaMemcpy(dev_index_max, max_index, numMat*sizeof(size_t), cudaMemcpyHostToDevice);
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
	//----------------------------------------------------
	// Varible to save the top BlockSize parts of 1K value
	//----------------------------------------------------
	float *max_value,*dev_max_value;
	cudaMalloc(&dev_max_value, BlockSize*TOP_T*sizeof(float));
	max_value = (float*)malloc(BlockSize*TOP_T*sizeof(float));
	printf(">> Laoding data comleted!\n");
	//-----------------
	// Invoke kernel
	//-----------------
	printf(">> Starting exact search!\n");
	cudaEvent_t start, stop;
	float elapsedTime = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	GetMaxValue<<<BlockSize,threadsPerBlock>>>(dev_data, \
											   numMat, \
											   dev_index_max, \
											   dim, \
											   dev_max_value);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf(">> GetMaxValue costs %f\n", elapsedTime);
	cudaDeviceSynchronize();
	//-------------------------------------------------
	// merge the BlockSize of TOP_T values into one
	//-------------------------------------------------
	int *potIdxBlock;
	cudaMalloc(&potIdxBlock, BlockSize*sizeof(int));
	cudaEventRecord(&start, 0);
	getPot<<<BlockSize,1>>>(dev_max_value,potIdxBlock);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf(">> getPot costs %f\n", elapsedTime);
	
	//--------------------------------------------------
	// Copy result from device memory to host memory
	//--------------------------------------------------
	int *h_potIdxBlock;
	h_potIdxBlock = (int*)malloc(BlockSize*sizeof(int));
	cudaMemcpy(h_potIdxBlock,potIdxBlock,BlockSize*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(max_value,dev_max_value,BlockSize*TOP_T*sizeof(float),cudaMemcpyDeviceToHost);
	//------------------------------------
	// Get the top 1K of max_value
	//------------------------------------
	float *top_t = (float*)malloc(TOP_T*sizeof(float));
	mergeSort(max_value,top_t,h_potIdxBlock);
	/*
	for(int i = 1; i < BlockSize; ++i){
		for(int j = 0; j < TOP_T; ++j){
			if(max_value[i*TOP_T + j] > max_value[TOP_T-1]){
				//insert
				doInsert(max_value[i*TOP_T + j], max_value, TOP_T);
			}
		}
	}
	*/
	//----------------------------
	// write the top-1K to file
	//----------------------------
	FILE *fp_out = fopen(resultfile, "w");
	for (int i = 0; i < TOP_T; ++i){
		fprintf(fp_out,"%f\n",top_t[i]);
		//fprintf(fp_out,"%f\n",max_value[i]);
	}
	fclose(fp_out);
	//-----------------------------------
	// Free device memory and host memory
	//-----------------------------------
	for (int i = 0; i < numMat; ++i){
		cudaFree(h_data[i].element);
		free(data[i].element); 
	}
	cudaFree(dev_index_max);
	cudaFree(dev_data);
	cudaFree(dev_max_value);
	cudaFree(potIdxBlock);
	free(line);
	free(max_index);
	free(data);
	free(h_data);
	free(max_value);
	free(h_potIdxBlock);
	free(top_t);

	return 0;
}

__global__ void GetMaxValue(Matrix *dev_data,\
							int numMat, \
							size_t *max_index, \
							int d, \
							float *max_value){
	// shared memory to save threadsPerBlock parts of top 1K value
	__shared__ float cache[threadsPerBlock][TOP_T];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;


	float *thread_max_value;
	float *mul_value;
	float tmp_max;
	struct cuSubIndex subIndex(numMat, max_index);
	thread_max_value = (float*)malloc(TOP_T*sizeof(float));
	mul_value = (float*)malloc(d*sizeof(float));

	for(int i = 0; i < TOP_T; ++i){
		thread_max_value[i] = 0;
	}
	for(int i = 0; i < d; ++i){
		mul_value[i] = 1;
	}

	subIndex = subIndex + tid;
	while(!(subIndex.isDone)){
		for(int i = 0; i < numMat; ++i){
			for(int j = 0; j < d; ++j){
				mul_value[j] *= dev_data[i].element[d*(subIndex.index_cur[i]) + j];
			}
		}
		for(int i = 0; i < d; ++i){
			tmp_max += mul_value[i];
		}
		if(tmp_max > thread_max_value[TOP_T - 1]){
			doInsert(tmp_max,thread_max_value,TOP_T);
		}
		subIndex = subIndex + blockDim.x * gridDim.x;
	}
	// put the top 1K value of thread to global memory
	for(int i = 0; i < TOP_T; ++i){
		cache[threadIdx.x][i] = thread_max_value[i];
	}
	__syncthreads();

	int cacheIndex = blockDim.x/2;
	// merge two 1K value vector into one vector
	while(cacheIndex != 0){
		if(threadIdx.x < cacheIndex){
			for(int i = 0; i < TOP_T; ++i){
				if(cache[threadIdx.x + cacheIndex][i] > cache[threadIdx.x][TOP_T - 1]){
					doInsert(cache[threadIdx.x + cacheIndex][i],\
							&(cache[threadIdx.x][0]),TOP_T);
				}
			}
		}
		__syncthreads();
		cacheIndex /= 2;
	}
	// cache[0][:] is the final top 1K value
	// put it into global memory.
	// final BlockSize 1K value vector
	if(threadIdx.x == 0){
		for(int i = 0; i < TOP_T; ++i){
			max_value[TOP_T*blockIdx.x + i] = cache[0][i];
		}
	}
	free(thread_max_value);
	free(mul_value);
}


__global__ void getPot(float *src, int *potIdxBlock){
	double temp;
	int index = 0;
	bool doneSearch = false;
	size_t pos = blockIdx.x*TOP_T;
	float *potV = (float*)malloc(BlockSize*sizeof(float));
	for(int i = 0; i < TOP_T; ++i){
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
			// escape self beam
			if(m == blockIdx.x) continue; 
			// search in other beam
			for(int k = 0; k < TOP_T; ++k){
				// encounter a bigger value 
				if(temp < src[pos + k]){
					++index;
				}else{
					break;
				}
				// if this value is out of the top-t
				// we can finish searching
				if(index >= TOP_T){
					doneSearch = true;
					break;
				}
			}
		}
		// if beam[i] has no potential to be the top-t
		// then the number of elements have the potential
		// is i then record it;
		if(doneSearch){
			potIdxBlock[blockIdx.x] = i;
			break;
		}
  	}
  	
}

void mergeSort(float*src,float*dst, int *potIdxBlock){
  	int count = 0;
 	for(int m = 0; m < BlockSize; ++m){
		count += potIdxBlock[m];
	}
	float *potV = (float*)malloc(count*sizeof(float));
	int p = 0;
 	for(int m = 0; m < BlockSize; ++m){
		for(int i = 0 ; i < potIdxBlock[m]; ++i){
			potV[p] = src[m*TOP_T + i];
			++p;
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
	for(int i = 0; i < TOP_T; ++i){
		dst[i] = potV[i];
	}
	free(potV);
}