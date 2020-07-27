#ifndef __UTILS_H__
#define __UTILS_H__

#include <cmath>
#include <cstring>
#include <ctime>

#include "common.h"
/**
 * @biref compute the sign of x
 *
 */
int sgn(const double &x);
int sgn_foo(const double &x);
double sqr(double x);

double sigmoid(double x);

double timeDuration(clock_t start);

/**
 *  @brief using binary search to get s uint with given distribution
 *  @param[in] s : number of random results
 *  @param[in] *dst : the random numbers
 *  @param[in] *pdf : distribution that sums to 1
 */
void binary_search(size_t s, uint *dst, uint n, double *pdf);
/**
 *  @brief binary search
 *  @param[in] *a : the search vector
 *  @param[in] ub : the length of a
 *  @param[in] s : search value
 */
uint binary_search_once(double *a, uint ub, double s);
void sort_sample(size_t s, uint *i, uint *r, size_t *freq, uint m, uint n,
                 double *pdf, double sum_pdf);
void sort_sample(size_t s, uint *dst, uint n, double *p, double sum);
void vose_alias(size_t s, uint *dst, uint n, double *pdf, double sum_pdf);

void progressbar(double progress);

void clearprogressbar();
class ProgressBar {
 public:
  ProgressBar(int width);
  void update(double progress);
  void clear();

 private:
  int barWidth;
  bool isDrawn;
};

class Timer {
 public:
  Timer();
  void r_init(clock_t start);
  void r_samp(clock_t start);
  void r_score(clock_t start);
  void r_filter(clock_t start);

 public:
  double initialization;
  double sampling;
  double scoring;
  double filtering;
};

/**
 *  @brief counter for uint array[n]
 *
 *  each position in arry has maximum
 */
class IndexCounter {
 public:
  IndexCounter(uint n, uint *max);
  ~IndexCounter();
  /**
   *  @brief if all element meets maximum
   */
  bool isDone() { return done; };
  /**
   *  @brief reset to all zeros
   */
  bool reset();
  IndexCounter &operator++();
  /**
   *  @brief get current indexes
   */
  const uint *getIdx() { return current; };

 private:
  uint length;
  bool done;
  uint *current;
  const uint *maximum;
};

#endif /*__UTILS_H__*/
