/**
 * utils.h
 *
 * Created by Dimitrios Karageorgiou,
 *  for course "Parallel And Distributed Systems".
 *  Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
 *
 * This header file contains various utility functions, used throughout
 * current project.
 *
 * Functions defined in utils.h
 *  -matrix_t *array_to_matrix(double *a, int rows, int cols, int lda)
 *  -double *matrix_to_array(matrix_t *matrix, int ldm)
 *  -double get_elapsed_time(struct timeval start, struct timeval stop)
 *  -int nearest_two_pow(int n, int lower)
 */

#ifndef __utils_h__
#define __utils_h__


#include <sys/time.h>
#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Convert a matrix represented as linear array in column-major order,
 * to a matrix object.
 *
 * Parameters:
 *  -a : An array of size lda*cols, representing a matrix stored in
 *          column-major order.
 *  -rows : Number of rows contained in a.
 *  -cols : Number of columns contained in a.
 *  -lda : Leading dimension of a (usually lda = rows, unless there is
 *           some padding).
 *
 * Returns:
 *  A matrix object containing the values of provided array.
 */
matrix_t *array_to_matrix(double *a, int rows, int cols, int lda);

/**
 * Convert a matrix to array stored in a column-major order.
 *
 * Parameters:
 *  -matrix : A matrix object to serialize.
 *  -ldm : Leading dimension the returned array should has. It usually is
 *          the number of rows of the matrix, unless some padding is needed.
 *
 * Returns:
 *  An array of size lda*cols, containing the values of the matrix in
 *  column-major order.
 */
double *matrix_to_array(matrix_t *matrix, int ldm);

/**
 * Returns the elapsed time between two timestamps.
 *
 * Parameters:
 *  -start : Beggining timestamp of a time period.
 *  -stop : Ending timestamp of a time period.
 *
 * Returns:
 *  The time between two provided timestamps, in seconds.
 */
double get_elapsed_time(struct timeval start, struct timeval stop);

/**
 * Finds the nearest two power of the given number.
 *
 * Parameters:
 *  -n : Number to find its
 */
int nearest_two_pow(int n, int lower);


#ifdef __cplusplus
}
#endif

#endif
