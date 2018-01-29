/**
 * cuda_mean_shift.h
 */

#ifndef __cuda_mean_shift_h__
#define __cuda_mean_shift_h__


#include "matrix.h"

matrix_t *cuda_mean_shift(matrix_t *points, double h, double e, int verbose);

#endif
