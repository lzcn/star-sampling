#include "utils.h"

#include "mex.h"

int sgn(const double &x) { return (x < 0 ? -1 : 1); }
int sgn_foo(const double &x) { return (x < 0 ? -1 : 1); }
double sqr(double x) { return x * x; }

double sigmoid(double x) { return (double)1.0 / (1.0 + exp(-x)); }

double timeDuration(clock_t start) {
  return (double)(clock() - start) / (double)CLOCKS_PER_SEC;
}
void binary_search(size_t s, uint *dst, uint n, double *pdf) {
  double sum = pdf[n - 1];
  for (size_t i = 0; i < s; ++i) {
    double u = sum * ((double)rand() / (double)RAND_MAX);
    *(dst + i) = binary_search_once(pdf, n - 1, u);
  }
}
uint binary_search_once(double *a, uint ub, double s) {
  uint m;
  uint lb = 0;
  if (s < a[0]) return 0;
  while (lb < ub - 1) {
    m = (lb + ub) / 2;
    if (s < a[m])
      ub = m;
    else
      lb = m;
  }
  return (ub);
}
void vose_alias(size_t s, uint *dst, uint n, double *pdf, double sum_pdf) {
  double *scaled_prob = new double[n];
  double *table_prob = new double[n];
  uint *table_alias = new uint[n];
  uint *table_small = new uint[n];
  uint *table_large = new uint[n];
  uint small_index = 0;
  uint large_index = 0;
  /* stage 1: initialization */
  for (uint i = 0; i < n; ++i) {
    scaled_prob[i] = abs(*(pdf + i)) * n;
    if (scaled_prob[i] < sum_pdf) {
      table_small[small_index] = i;
      ++small_index;
    } else {
      table_large[large_index] = i;
      ++large_index;
    }
  }
  uint l, g;
  while (small_index != 0 && large_index != 0) {
    small_index -= 1;
    large_index -= 1;
    l = table_small[small_index];
    g = table_large[large_index];
    table_prob[l] = scaled_prob[l];
    table_alias[l] = g;
    scaled_prob[g] = (scaled_prob[g] + scaled_prob[l]) - sum_pdf;

    if (scaled_prob[g] < sum_pdf) {
      table_small[small_index] = g;
      ++small_index;
    } else {
      table_large[large_index] = g;
      ++large_index;
    }
  }
  while (large_index != 0) {
    large_index -= 1;
    table_prob[table_large[large_index]] = sum_pdf;
  }
  while (small_index != 0) {
    small_index -= 1;
    table_prob[table_small[small_index]] = sum_pdf;
  }
  /* stage 2: random sampling */
  double u;
  uint fair_die;
  for (size_t i = 0; i < s; ++i) {
    fair_die = rand() % n;
    u = sum_pdf * (double)rand() / (double)RAND_MAX;
    if (table_prob[fair_die] >= u) {
      *(dst + i) = fair_die;
    } else {
      *(dst + i) = table_alias[fair_die];
    }
  }
  delete[] table_prob;
  delete[] table_alias;
  delete[] scaled_prob;
  delete[] table_small;
  delete[] table_large;
}

void sort_sample(size_t s, uint *dst, uint n, double *p, double sum) {
  std::vector<double> rand_u;
  for (size_t i = 0; i < s; ++i) {
    rand_u.push_back(sum * ((double)rand() / (double)RAND_MAX));
  }
  sort(rand_u.begin(), rand_u.end());
  uint ind = 0;
  double prob_accum = abs(p[0]);
  for (size_t i = 0; i < s; ++i) {
    while ((rand_u[i] >= prob_accum) && (ind < (n - 1))) {
      prob_accum += abs(p[++ind]);
    }
    dst[i] = ind;
  }
}
void sort_sample(size_t s, uint *idxI, uint *idxR, size_t *freq, uint m, uint n,
                 double *pdf, double sum_pdf) {
  std::vector<double> rand_u;
  for (size_t i = 0; i < s; ++i) {
    rand_u.push_back(sum_pdf * ((double)rand() / (double)RAND_MAX));
  }
  // Sort the random values
  // It will be sorted according to k then i;
  sort(rand_u.begin(), rand_u.end());
  uint ind = 0;
  uint range = m * n;
  double sum_prob = abs(pdf[0]);
  for (size_t i = 0; i < s; ++i) {
    while ((rand_u[i] >= sum_prob) && (ind < (range - 1))) {
      sum_prob += abs(pdf[++ind]);
    }
    idxI[i] = ind % n;
    idxR[i] = ind / n;
    ++freq[idxR[i]];
  }
}

void progressbar(double progress) {
  int barWidth = 40;
  int pos = (int)(barWidth * progress);
  mexPrintf("[");
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos)
      mexPrintf("=");
    else if (i == pos)
      mexPrintf(">");
    else
      mexPrintf(" ");
  }
  mexPrintf("]");
  mexPrintf("%3d%%", int(progress * 100.0));
  mexEvalString("drawnow");
}

void clearprogressbar() {
  int barWidth = 46;
  for (int i = 0; i < barWidth; ++i) mexPrintf("\b");
  mexEvalString("drawnow");
}

ProgressBar::ProgressBar(int width) : barWidth{width}, isDrawn{false} {}

void ProgressBar::update(double progress) {
  int pos = (int)(barWidth * progress);
  mexPrintf("[");
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos)
      mexPrintf("=");
    else if (i == pos)
      mexPrintf(">");
    else
      mexPrintf(" ");
  }
  mexPrintf("]");
  mexPrintf("%3d%%", int(progress * 100.0));
  mexEvalString("drawnow");
  isDrawn = true;
}

void ProgressBar::clear() {
  if (isDrawn) {
    for (int i = 0; i < barWidth + 4; ++i) mexPrintf("\b");
    mexEvalString("drawnow");
  }
}

Timer::Timer()
    : initialization{0.0}, sampling{0.0}, scoring{0.0}, filtering{0.0} {}

void Timer::r_init(clock_t start) {
  initialization += (double)(clock() - start) / CLOCKS_PER_SEC;
}
void Timer::r_samp(clock_t start) {
  sampling += (double)(clock() - start) / CLOCKS_PER_SEC;
}
void Timer::r_score(clock_t start) {
  scoring += (double)(clock() - start) / CLOCKS_PER_SEC;
}
void Timer::r_filter(clock_t start) {
  filtering += (double)(clock() - start) / CLOCKS_PER_SEC;
}

IndexCounter::IndexCounter(uint n, uint *max) {
  length = n;
  maximum = max;
  done = false;
  current = (uint *)malloc((n + 1) * sizeof(uint));
  memset(current, 0, (n + 1) * sizeof(uint));
}
IndexCounter::~IndexCounter() { free(current); }

bool IndexCounter::reset() {
  done = false;
  memset(current, 0, (length + 1) * sizeof(uint));
  return true;
}

IndexCounter &IndexCounter::operator++() {
  ++current[0];
  for (uint i = 0; i < length; ++i) {
    if (current[i] < maximum[i]) {
      return *this;
    } else {
      current[i] = 0;
      ++current[i + 1];
    }
  }
  if (current[length] >= 1) {
    done = true;
  }
  return *this;
}

