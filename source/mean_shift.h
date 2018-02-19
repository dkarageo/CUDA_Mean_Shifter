/**
 * mean_shift.h
 *
 * Created by Dimitrios Karageorgiou,
 *  for course "Parallel And Distributed Systems".
 *  Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
 *
 * This header file provides routines to execute mean shift on a dataset.
 *
 * Functions defined in mean_shift.h:
 *  -matrix_t *mean_shift(matrix_t *points, double h, double e)
 *  -void shift_point(matrix_t *shifted, matrix_t *m, int i, matrix_t *original,
 *                 double h)
 */

#ifndef __matrix_h__
#define __matrix_h__


#include "matrix.h"


/**
 * Executes mean shift on given dataset, based on a guassian kernel.
 *
 * Convergence check is based on the frob-norm of mean shift vector.
 *
 * Parameters:
 *  -points : A matrix containing a dataset (rows correspond to points and
 *          columns to their coordinates) to be shifted.
 *  -h : Scalar to be used by gaussian kernel.
 *  -e : The edge value for frob-norm of mean shift vector, below which the
 *          accuracy of convergence is considered acceptable.
 *
 * Returns:
 *  A new matrix containing the shifted points.
 */
matrix_t *mean_shift(matrix_t *points, double h, double e);

/**
 * Shifts given point.
 *
 * Shifting of the point is done by utilizing a gaussian kernel, based on the
 * distance from original point.
 *
 * Parameters:
 *  -shifted : A matrix containing the current instance of shifted points. Upon
 *          successful shifting, the shifted point will be updated on this
 *          matrix.
 *  -m : A matrix where mean shift vector will be return.
 *  -i : The index of point to be shifted in shifted matrix and also for the
 *          returned mean shift value in m matrix.
 *  -original : A matrix containing the original positions of the points.
 *  -h : Scalar to be used by gaussian kernel. 
 */
void shift_point(matrix_t *shifted, matrix_t *m, int i, matrix_t *original,
                 double h);


#endif
