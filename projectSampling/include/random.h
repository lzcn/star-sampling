#ifndef __RANDOM_H__
#define __RANDOM_H__

#include <cstdlib>
#include <cmath>
#include <cassert>
#include "util.h"

namespace rnd{
	// normal distribution
	double randn();
	double randn(double mean, double stdev);
	// uniform distribution
	double randu();
	// exponential distribution
	double exprnd();	
	// gamma distribution
	double gamrnd(double alpha, double beta);
	double gamrnd(double alpha);
	// Bernoulli distribution
	bool binornd(double p);
	// binary search
	int binary_search(size_t s, size_t*dst, size_t n, double*pdf);
	// sampling via alias method
	int alias(size_t s, size_t *dst, size_t n, double *pdf, double sum);
	// sampling via sorting
	int sample_sort(size_t s, size_t*dst, size_t n, double*pdf, double sum);
}

double rnd::gamrnd(double alpha) {
	assert(alpha > 0);
	if (alpha < 1.0) {
		double u;
		do {
			u = randu();
		} while (u == 0.0);
		return gamrnd(alpha + 1.0) * pow(u, 1.0 / alpha);
	} else {
		// Marsaglia and Tsang: A Simple Method for Generating Gamma Variables
		double d,c,x,v,u;
		d = alpha - 1.0/3.0;
		c = 1.0 / std::sqrt(9.0 * d);
		do {
			do {
				x = randn();
				v = 1.0 + c*x;
			} while (v <= 0.0);
			v = v * v * v;
			u = randu();
		} while ( 
			(u >= (1.0 - 0.0331 * (x*x) * (x*x)))
			 && (log(u) >= (0.5 * x * x + d * (1.0 - v + std::log(v))))
			 );
		return d*v;
	}
}

double rnd::gamrnd(double alpha, double beta){
	return (gamrnd(alpha) / beta);
}

double rnd::randn(){
	// Joseph L. Leva: A fast normal Random number generator
	double u,v, x, y, Q;
	do {
		do {
			u = randu();
		} while (u == 0.0); 
		v = 1.7156 * (randu() - 0.5);
		x = u - 0.449871;
		y = std::abs(v) + 0.386595;
		Q = x*x + y*(0.19600*y-0.25472*x);
		if (Q < 0.27597) { break; }
	} while ((Q > 0.27846) || ((v*v) > (-4.0*u*u*std::log(u)))); 
	return v / u;
}

double rnd::randn(double mean, double stdev) {
	if ((stdev == 0.0) || (std::isnan(stdev))) {
		return mean;
	} else {
		return mean + stdev*randn();
	}
}

double rnd::randu() {
	return randn()/((double)RAND_MAX + 1);
}

double rnd::exprnd() {
	return -std::log(1-randu());
}

bool rnd::binornd(double p) {
	return (randu() < p);
}

int rnd::alias(size_t s, size_t *dst, size_t n, double *pdf,double sum){
	double *scaled_prob = new double[n];
	double *table_prob = new double[n];
	size_t *table_alias = new size_t[n];
	size_t *table_small = new size_t[n];
	size_t *table_large = new size_t[n];
	size_t small_index = 0;
	size_t large_index = 0;
	/* stage 1: initialization */
	for (size_t i = 0; i < n; ++i){
		scaled_prob[i] = abs(*(pdf+i)) * n;
		if ( scaled_prob[i] < sum ){
			table_small[small_index] = i;
			++small_index;
		}else{
			table_large[large_index] = i;
			++large_index;
		}
	}
	size_t l,g;
	while(small_index != 0 && large_index != 0){
		small_index -= 1;
		large_index -= 1;
		l = table_small[small_index];
		g = table_large[large_index];
		table_prob[l] = scaled_prob[l];
		table_alias[l] = g;
		scaled_prob[g] = (scaled_prob[g] + scaled_prob[l]) - sum;

		if (scaled_prob[g] < sum){
			table_small[small_index] = g;
			++small_index;
		}else{
			table_large[large_index] = g;
			++large_index;
		}
	}
	while(large_index != 0){
		large_index -= 1;
		table_prob[table_large[large_index]] = sum;
	}
	while(small_index != 0){
		small_index -= 1;
		table_prob[table_small[small_index]] = sum;
	}
	/* stage 2: random sampling */
	double u;
	size_t fair_die;
	for (size_t i = 0; i < s; ++i ){
		fair_die = rand() % n;
		u = sum*(double)rand()/(double)RAND_MAX;
		if (table_prob[fair_die] >= u){
			*(dst + i) = fair_die;
		}else{
			*(dst + i) = table_alias[fair_die];
		}
	}
	delete []table_prob;
	delete []table_alias;
	delete []scaled_prob;
	delete []table_small;
	delete []table_large;
	return 1;
}

int rnd::sample_sort(size_t s, size_t*dst, size_t n, double*pdf, double sum){
	std::vector<double> rand_u;
	for (size_t i = 0; i < s; ++i){
		rand_u.push_back(sum*((double)randn()/(double)RAND_MAX));
	}
	sort(rand_u.begin(),rand_u.end());
	size_t ind = 0;
	double prob_accum = abs(pdf[0]);
	for (size_t i = 0; i < s; ++i){
		while((rand_u[i] >= prob_accum) && (ind < (n-1))){
			prob_accum += abs(pdf[++ind]);
		}
		dst[i] = ind;
	}
	return 1;
}

int rnd::binary_search(size_t s, size_t*dst, size_t n, double*pdf){
	double *prob_accum = (double*)malloc(n*sizeof(double));
	double sum = 0.0;
	for(size_t i = 0; i < n; ++i){
		sum += abs(pdf[i]);
		prob_accum[i] = sum;
	}
	for(size_t i = 0; i < s; ++i){
		double u = sum*((double)randn()/(double)RAND_MAX);
		int d = 0;
		for(; u >= prob_accum[d] && d < n; ++d);
		if (d < n)
			dst[s] = d;
		else
			dst[s] = n - 1;
	}
	free(prob_accum);
	return 1;
}
#endif /*__RANDOM_H__*/
