/**
 * algebra.h
 *
 * Created by Dimitrios Karageorgiou,
 *  for course "Parallel And Distributed Systems".
 *  Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
 *
 * This header file provides routines that implement linear algebra functions.
 *
 * Functions defined in algebra.h:
 *  -double norm(matrix_t *matrix)
 *  -double euclidian_dist(matrix_t *m1, int32_t p1, matrix_t *m2, int32_t p2)
 *  -double gaussian_kernel(double dist, double scalar)
 */

#ifndef __algebra_h__
#define __algebra_h__

#include "matrix.h"


/**
 * Calculates the Frobenius Norm of a matrix.
 */
double norm(matrix_t *matrix);

double euclidian_dist(matrix_t *m1, int32_t p1, matrix_t *m2, int32_t p2);

double gaussian_kernel(double dist, double scalar);


#endif
