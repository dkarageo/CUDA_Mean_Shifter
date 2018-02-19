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
 *
 * Parameters:
 *  -matrix : A matrix whose frobenius norm to be calculated.
 *
 * Returns:
 *  Frobenius norm of the given matrix.
 */
double norm(matrix_t *matrix);

/**
 * Calculates the euclidian distance between two points.
 *
 * Parameters:
 *  -m1 : A matrix containing the first point.
 *  -p1 : Index of first point in m1.
 *  -m2 : A matrix containing the second point.
 *  -p2 : Index of second point in m2.
 *
 * Returns:
 *  The euclidian distance between given points.
 */
double euclidian_dist(matrix_t *m1, int32_t p1, matrix_t *m2, int32_t p2);

/**
 * Calculates the gaussian kernel.
 *
 * Parameters:
 *  -dist : The value (possibly a distance) to calculate its gaussian kernel.
 *  -scalar : Scalar value to be used in gaussian's kernel calculation.
 *
 * Returns:
 *  The gaussian kernel value.
 */
double gaussian_kernel(double dist, double scalar);


#endif
