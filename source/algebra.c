/**
 * algebra.c
 *
 * Created by Dimitrios Karageorgiou,
 *  for course "Parallel And Distributed Systems".
 *  Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
 *
 * This file provides an implementation for routines defined in algebra.h.
 */

#include <math.h>

#include "svd/svd_double.h"
#include "algebra.h"


double norm(matrix_t *matrix)
{
    // Create a copy of provided matrix, since dsvd() is gonna replace
    // it with U.
    matrix_t *copy = matrix_create_copy(matrix);
    double **copy_array = matrix_to_2d_array(copy);

    int32_t rows = matrix_get_rows(copy);
    int32_t cols = matrix_get_cols(copy);

    // Create a vector that will contain the singular values.
    double *w = (double *) malloc(sizeof(double) * rows);
    for (int i = 0; i < rows; i++) w[i] = 0.0;

    matrix_t *v = matrix_create(rows, cols);
    double **v_array = matrix_to_2d_array(v);

    dsvd(copy_array, rows, cols, w, v_array);

    // Calculate frob norm.
    double frob = 0.0;
    for (int i = 0; i < cols; i++) frob += w[i]*w[i];
    frob = sqrt(frob);

    matrix_destroy(copy);
    matrix_destroy(v);
    free(w);

    return frob;
}

double euclidian_dist(matrix_t *m1, int32_t p1, matrix_t *m2, int32_t p2)
{
    double dist = 0;

    for (int j = 0; j < matrix_get_cols(m1); j++) {
        dist += pow(matrix_get_cell(m1, p1, j) - matrix_get_cell(m2, p2, j), 2.0);
    }

    return sqrt(dist);
}

double gaussian_kernel(double dist, double scalar)
{
    return exp(-dist / (2 * powf(scalar, 2.0)));
}
