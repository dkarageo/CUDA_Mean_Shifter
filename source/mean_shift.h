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

#include "matrix.h"


matrix_t *mean_shift(matrix_t *points, double h, double e);
void shift_point(matrix_t *shifted, matrix_t *m, int i, matrix_t *original,
                 double h);
