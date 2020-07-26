#ifndef __RANDOM_H__
#define __RANDOM_H__

#include <cassert>
#include <cmath>
#include <cstdlib>

#include "common.h"
namespace rnd {
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
int binary_search(size_t s, size_t *dst, size_t n, double *pdf);
// sampling via alias method
int alias(size_t s, size_t *dst, size_t n, double *pdf, double sum);
// sampling via sorting
int sample_sort(size_t s, size_t *dst, size_t n, double *pdf, double sum);
}  // namespace rnd

#endif /*__RANDOM_H__*/
