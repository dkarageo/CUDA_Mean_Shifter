/**
 * cuda_mean_shift_nosm.cu
 *
 * Created by Dimitrios Karageorgiou,
 *  for course "Parallel And Distributed Systems".
 *  Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
 *
 * This file provides an implementation for the routines defined in
 * cuda_mean_shift_nosm.h.
 */

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <sys/time.h>
#include "algebra.h"
#include "utils.h"
#include "matrix.h"
extern "C" {
#include "cuda_algebra.h"
};

#define BLOCK_SIZE 64


__global__ void shift_point_nosm_kernel(
                            int rows,
                            int cols,
                            int lda,
                            double *d_shifted,
                            double *d_m,
                            double *d_original,
                            double h,
                            double *d_n_work);


extern "C"
matrix_t *cuda_mean_shift_nosm(matrix_t *points, double h, double e, int verbose)
{
    int rows = matrix_get_rows(points);
    int cols = matrix_get_cols(points);
    int ldp = rows;  // Leading dimension of points matrix.
    size_t size = sizeof(double) * ldp * cols;  // Final padded size of matrix.

    // Matrices on host.
    matrix_t *shifted;

    // Matrices on device.
    double *d_original;    // Original points that should not get altered (rows*cols)
    double *d_shifted;     // Points after mean shift (rows*cols).
    double *d_mean_shift;  // Mean-shift vector (d_original - d_shifted) (rows*cols).
    double *d_work1;       // Temp workspace 1. (rows*cols).

    // Status codes.
    cudaError_t cudaStat1;
    cudaError_t cudaStat2;
    cudaError_t cudaStat3;
    cudaError_t cudaStat4;

    // Timestamps for calculating pure execution time.
    struct timeval start, stop, norm_start, norm_stop;

    if (cols > rows ) {
        printf("ERROR: cuda_mean_shift() failed. rows should be gt cols.");
        return NULL;
    }

    // Allocate memory on device.
    cudaStat1 = cudaMalloc(&d_original, size);
    cudaStat2 = cudaMalloc(&d_shifted, size);
    cudaStat3 = cudaMalloc(&d_mean_shift, size);
    cudaStat4 = cudaMalloc(&d_work1, size);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    // Copy given matrix to device.
    double *orig = matrix_to_array(points, ldp);
    cudaStat1 = cudaMemcpy(d_original, orig, size, cudaMemcpyHostToDevice);
    free(orig);
    assert(cudaSuccess == cudaStat1);

    gettimeofday(&start, NULL);  // Start calculating time.

    // Initial assumption for the shifted points are the original points.
    cudaStat1 = cudaMemcpy(
            d_shifted, d_original, size, cudaMemcpyDeviceToDevice);
    assert(cudaSuccess == cudaStat1);

    // Calculate dims of each thread block.
    int block_width = 1;  // Width should be fixed to 1.
    int block_height = BLOCK_SIZE;
    dim3 block_dim(block_width, block_height);

    // Calculate dims of blocks grid. It's width will always be 1.
    int grid_height = rows / block_height;
    if ((rows % block_height) > 0) grid_height++;
    dim3 grid_dim(1, grid_height);  // Grid should not contain more than 1
                                    // horizontal block.

    int iterations = 1;      // Count the iterations till convergence.
    double error = e * 2;    // Error after each iteration.
    double norm_time = 0.0;  // Calculate time spent on frob-norm calculation.

    // Repeat shifting until convergence.
    while(error > e) {

        shift_point_nosm_kernel<<<grid_dim, block_dim>>>(
                rows,
                cols,
                ldp,
                d_shifted,
                d_mean_shift,
                d_original,
                h,
                d_work1);

        cudaDeviceSynchronize();

        gettimeofday(&norm_start, NULL);  // Calculate norm time.

        error = cuda_norm(d_mean_shift, rows, cols, ldp);

        gettimeofday(&norm_stop, NULL);
        norm_time += get_elapsed_time(norm_start, norm_stop);

        if (verbose) printf("Iteration %d - Error: %f\n", iterations, error);
        iterations++;
    }

    gettimeofday(&stop, NULL);  // Stop calculating time.
    if (verbose) {
        printf("CUDA (No SM) Frob-norm Calc Time (cusolver): %.3f secs\n", norm_time);
        printf("CUDA (No SM) Mean Shift Pure Time: %.3f secs\n",
               get_elapsed_time(start, stop) - norm_time);
        printf("CUDA (No SM) Mean Shift Total Time: %.3f secs\n",
               get_elapsed_time(start, stop));
    }

    // Copy back shifted points from device to host.
    double *shifted_array = (double *) malloc(size);
    assert(NULL != shifted_array);
    cudaStat1 = cudaMemcpy(
            shifted_array, d_shifted, size, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    shifted = array_to_matrix(shifted_array, rows, cols, ldp);
    free(shifted_array);

    // Release resources.
    cudaFree(&d_original);
    cudaFree(&d_shifted);
    cudaFree(&d_mean_shift);
    cudaFree(&d_work1);

    return shifted;
}

__global__ void shift_point_nosm_kernel(
                            int rows,
                            int cols,
                            int lda,
                            double *d_shifted,
                            double *d_m,
                            double *d_original,
                            double h,
                            double *d_n_work)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows) {
        // Initialize nominator and denominator to 0.
        // d_n_work utilized as workspace for nominators.
        for (int k = 0; k < cols; k++) d_n_work[i + k*lda] = 0.0;
        double denom = 0.0;

        // Compute the kernels between current point and all original ones.
        for (int j = 0; j < rows; j++) {
            // Calculate the square distance between current points pair.
            double sq_dist = 0.0;
            for (int k = 0; k < cols; k++) {
                double d = d_shifted[i + k*lda] - d_original[j + k*lda];
                sq_dist += d * d;
            }

            // Ignore points on a distance greater than h.
            if (sq_dist < h*h) {
                double kernel = exp(-sq_dist / (2*h*h));  // Calc kernel.

                // Add original[j]*kernel to nominator.
                for (int k = 0; k < cols; k++)
                    d_n_work[i + lda*k] += d_original[j + k*lda] * kernel;

                // Add kernel to denominator.
                denom += kernel;
            }
        }

        // In case no distance found lesser than scaler, just don't update this point.
        for (int k = 0; k < cols; k++) {
            // Calculate the new vector of actual shifted point.
            double new_cord = d_n_work[i + k*lda] / denom;

            // Calculate mean shift vector for current point
            d_m[i + k*lda] = new_cord - d_shifted[i + k*lda];

            // Finally, update the previously shifted vector with the new one.
            d_shifted[i + k*lda] = new_cord;
        }
    }
}
