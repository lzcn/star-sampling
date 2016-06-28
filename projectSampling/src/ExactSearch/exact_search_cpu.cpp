#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>


// find top 1K value
#define TOP1K 1000



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
// SubIndex is used for loop
/* 
	data: 
		index_cur: indicate the current index for 
		index_max: each element is the number vector
				of correspongding maxtirx
	method: 
		+ :add t to current index
*/
//------------------------------------------------
struct SubIndex
{
	int numMat;
	size_t *index_cur;
	const size_t *index_max;
	bool isDone;
	SubIndex(int n, size_t *max) :numMat(n){
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
	~SubIndex(){
		//free(index_max);
		free(index_cur);
	}
	SubIndex& operator+(const size_t step){
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
		return *this;
	}
};


// data struct : matrix
typedef struct 
{
	size_t dim;
	size_t num;
	float *element;
}Matrix;

int doInsert(float value, float* toInsert, int num){
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

void GetMaxValue(Matrix *data, \
						int numMat, \
						size_t *max_index, \
						int d, \
						float*max_value);

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
	printf(">> Laoding data comleted!\n");
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
	//-------------------------------------------------
	//
	//-------------------------------------------------
	printf(">> Starting exact search!\n");
	float *max_value;
	max_value = (float*)malloc(TOP1K*sizeof(float));
	//-----------------
	// Invoke kernel
	//-----------------
	GetMaxValue(data, \
				numMat, \
				max_index, \
				dim, \
				max_value);

	//----------------------------
	// write the top-1K to file
	//----------------------------
	FILE *fp_out = fopen(resultfile, "w");
	for (int i = 0; i < TOP1K; ++i){
		fprintf(fp_out,"%f\n",max_value[i]);
	}
	fclose(fp_out);
	//-----------------------------------
	// Free device memory and host memory
	//-----------------------------------
	for (int i = 0; i < numMat; ++i){
		free(data[i].element); 
	}
	free(line);
	free(max_index);
	free(data);
	free(max_value);

	return 0;
}

__global__ void GetMaxValue(Matrix *data,\
							int numMat, \
							size_t *max_index, \
							int d, \
							float *max_value){
	// shared memory to save threadsPerBlock parts of top 1K value

	float *mul_value;
	float tmp_max;
	struct SubIndex subIndex(numMat, max_index);
	mul_value = (float*)malloc(d*sizeof(float));

	for(int i = 0; i < TOP1K; ++i){
		max_value[i] = 0;
	}
	for(int i = 0; i < d; ++i){
		mul_value[i] = 1;
	}

	while(!(subIndex.isDone)){
		for(int i = 0; i < numMat; ++i){
			for(int j = 0; j < d; ++j){
				mul_value[j] *= data[i].element[d*(subIndex.index_cur[i]) + j];
			}
		}
		for(int i = 0; i < d; ++i){
			tmp_max += mul_value[i];
		}
		if(tmp_max > max_value[TOP1K - 1]){
			doInsert(tmp_max,max_value,TOP1K);
		}
		subIndex = subIndex + 1;
	}
	free(mul_value);
}
