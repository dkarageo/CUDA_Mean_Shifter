/**
 * mean_shift.c
 *
 * Created by Dimitrios Karageorgiou,
 *  for course "Parallel And Distributed Systems".
 *  Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
 *
 * This file provides an implementation for routines defined in mean_shift.h
 */

#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include "matrix.h"
#include "algebra.h"
#include "utils.h"
#include "mean_shift.h"


matrix_t *mean_shift(matrix_t *points, double h, double e)
{
    // Initial assumption for the shifted points is the initial points unshifted.
    matrix_t *shifted = matrix_create_copy(points);

    // Initialize the mean shift vector to a value that will always pass the
    // first check on while loop. That value can be h.
    matrix_t *m = matrix_create(matrix_get_rows(shifted),
                                matrix_get_cols(shifted));
    matrix_fill(m, h);

    int iter = 1;
    double error = 2 * e;
    struct timeval norm_start, norm_stop;
    double norm_time = 0.0;

    // Repeat shifting until convergence.
    while(error > e) {
        int rows = matrix_get_rows(shifted);

        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            shift_point(shifted, m, i, points, h);
        }

        gettimeofday(&norm_start, NULL);
        error = norm(m);
        gettimeofday(&norm_stop, NULL);
        norm_time += get_elapsed_time(norm_start, norm_stop);

        printf("Iteration %d - Error: %f\n", iter, error);
        iter++;
    }

    printf("CPU Frob-norm Calc Time: %.3f secs\n", norm_time);

    matrix_destroy(m);
    return shifted;
}

void shift_point(matrix_t *shifted, matrix_t *m, int i, matrix_t *original,
                 double h)
{
    matrix_t *sum_nom = matrix_create(1, matrix_get_cols(shifted));
    matrix_fill(sum_nom, 0.0);
    double sum_denom = 0.0;

    // Calculate the sums both of nominator and denominator.
    for (int j = 0; j < matrix_get_rows(original); j++) {
        // Calculate the square of distance between current points pair.
        double sq_dist = euclidian_dist(shifted, i, original, j);

        if (sq_dist < h) {
            sq_dist = pow(sq_dist, 2.0);
            double kernel = gaussian_kernel(sq_dist, h);
            matrix_t *kx = matrix_row_num_mul(original, j, kernel, NULL, 0);
            matrix_row_add(sum_nom, 0, kx, 0, sum_nom, 0);
            matrix_destroy(kx);
            sum_denom += kernel;
        }
    }

    // Calculate the vector of actual shifted point.
    matrix_t *new_shifted_p = matrix_row_num_div(sum_nom, 0, sum_denom, NULL, 0);

    // Calculate mean shift vector for current point.
    matrix_row_sub(new_shifted_p, 0, shifted, i, m, i);

    // Finally, update the previously shifted vector with the new one.
    matrix_set_row(shifted, i, new_shifted_p, 0);

    matrix_destroy(sum_nom);
    matrix_destroy(new_shifted_p);
}
