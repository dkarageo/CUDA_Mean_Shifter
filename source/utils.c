/**
 * utils.c
 *
 * Created by Dimitrios Karageorgiou,
 *  for course "Parallel And Distributed Systems".
 *  Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
 *
 * This file provides an implementation for routines defined in utils.h 
 */

#include <sys/time.h>
#include <assert.h>
#include "matrix.h"


matrix_t *array_to_matrix(double *a, int rows, int cols, int lda)
{
    matrix_t *matrix = matrix_create((int32_t) rows, (int32_t) cols);
    assert(matrix != NULL);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix_set_cell(matrix, i, j, a[i + j*lda]);
        }
    }

    return matrix;
}

double *matrix_to_array(matrix_t *matrix, int ldm)
{
    int rows = matrix_get_rows(matrix);
    int cols = matrix_get_cols(matrix);

    // Leading dimension cannot be lower than any dim of the matrix.
    if (ldm < rows) ldm = rows;

    double *a = (double *) malloc(sizeof(double)*ldm*cols);
    assert(a != NULL);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            a[i + j*ldm] = matrix_get_cell(matrix, i, j);
        }
    }

    return a;
}

double get_elapsed_time(struct timeval start, struct timeval stop)
{
    double elapsed_time = (stop.tv_sec - start.tv_sec) * 1.0;
    elapsed_time += (stop.tv_usec - start.tv_usec) / 1000000.0;
    return elapsed_time;
}

int nearest_two_pow(int n, int lower)
{
    if (!n) return n;  // (0 == 2^0)

    int x = 1;
    while(x < n) x <<= 1;
    if (lower) x >>= 1;

    return x;
}
