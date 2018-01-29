#ifndef __algebra_h__
#define __algebra_h__

#include "matrix.h"


double norm(matrix_t *matrix);

double euclidian_dist(matrix_t *m1, int32_t p1, matrix_t *m2, int32_t p2);

double gaussian_kernel(double dist, double scalar);


#endif
