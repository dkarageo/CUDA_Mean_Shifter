/**
 * cuda_mean_shift.h
 *
 * Created by Dimitrios Karageorgiou,
 *  for course "Parallel And Distributed Systems".
 *  Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
 *
 * This header file provides routines to execute mean shifting on a dataset,
 * by utilizing cuda enabled GPUs and their Shared Memory feature.
 *
 * Functions defined in cuda_mean_shift.h:
 *  -matrix_t *cuda_mean_shift(matrix_t *points, double h, double e, int verbose)
 */

#ifndef __cuda_mean_shift_h__
#define __cuda_mean_shift_h__


#include "matrix.h"


/**
 * Executes mean shifting on provided points by utilizing CUDA enabled GPU.
 *
 * Shared Memory feature of CUDA enabled GPUs is utilized.
 *
 * Parameters:
 *  -points : A matrix containing the points to be shifted.
 *  -h : Scalar used by mean shift algorithm.
 *  -e : Requested precision for mean shift calculation.
 *  -verbose: A boolean value that when set to 0, no printing of calculation
 *          time is done. When set to 1, various times of calculation stages
 *          are printed to stdout.
 *
 * Returns:
 *  A matrix containing the shifted points, with the requested precision.
 */
matrix_t *cuda_mean_shift(matrix_t *points, double h, double e, int verbose);

#endif
