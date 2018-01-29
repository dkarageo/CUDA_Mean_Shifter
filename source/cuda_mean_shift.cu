/**
 * cuda_mean_shift.cu
 */

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <sys/time.h>
#include "algebra.h"
#include "matrix.h"
#include "utils.h"
extern "C" {
#include "cuda_algebra.h"
};

#define BLOCK_SIZE 128
#define SHARED_COLS 5


__global__ void shift_point_kernel(
                            int rows,
                            int cols,
                            int lda,
                            double *d_shifted,
                            double *d_m,
                            double *d_original,
                            double h,
                            double *d_n_work);

matrix_t *array_to_matrix(double *a, int rows, int cols, int lda);
double *matrix_to_array(matrix_t *matrix, int ldm);
double get_elapsed_time(struct timeval start, struct timeval stop);
int nearest_two_pow(int n, int lower);


extern "C"
matrix_t *cuda_mean_shift(matrix_t *points, double h, double e, int verbose)
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

    // Calculate dims of each thread block. It's size cannot exceed BLOCK_SIZE.
    // On matrices with less than 2000 rows, try to create more than
    // 2000 threads by increasing width of block.
    int multiplier = 1;  // Least needed block width to exceed 10000 threads.
    if (rows < 2000) multiplier = 2000 / rows;

    int block_width = nearest_two_pow(multiplier, 0);  // Width should be two's pow.
    if (block_width > BLOCK_SIZE) block_width = BLOCK_SIZE;  // And no more than BLOCK_SIZE.
    int block_height = BLOCK_SIZE / block_width;
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

        shift_point_kernel<<<grid_dim, block_dim>>>(
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
        printf("CUDA Frob-norm Calc Time (cusolver): %.3f secs\n", norm_time);
        printf("CUDA Mean Shift Pure Time: %.3f secs\n",
               get_elapsed_time(start, stop) - norm_time);
        printf("CUDA Mean Shift Total Time: %.3f secs\n",
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
    // cudaFree(&d_work2);

    return shifted;
}

__global__ void shift_point_kernel(
                            int rows,
                            int cols,
                            int lda,
                            double *d_shifted,
                            double *d_m,
                            double *d_original,
                            double h,
                            double *d_n_work)
{
    // Get block (i, cord) in d_shifted associated with current thread.
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int cord = threadIdx.x;  // Only one block is expected horizontally.

    // Initialize nominator and denominator to 0.
    // d_n_work utilized as workspace for nominators.
    if (i < rows)
        for (int k = cord; k < cols; k += blockDim.x)
            d_n_work[i + k*lda] = 0.0;
    double denom = 0.0;

    // Divide the d_original matrix, into blocks that can fit in shared memory.
    int chunks_y = rows / BLOCK_SIZE;
    int chunks_x = cols / SHARED_COLS;
    if (rows % BLOCK_SIZE > 0) chunks_y++;
    if (cols % SHARED_COLS > 0) chunks_x++;

    // Allocate shared memory.
    __shared__ double s_orig[BLOCK_SIZE][SHARED_COLS];   // Chunk of d_original.
    __shared__ double nom_red[BLOCK_SIZE][SHARED_COLS];  // Workspace for nom sum reduction.
    __shared__ double denom_red[BLOCK_SIZE];  // Workspace for denom sum reduction.

    // Allocate accumulators for intermediate values of distance and kernels,
    // since calculation will be done in many column-based iterations.
    // BLOCK_SIZE is the maximum possible size that may be utilized.
    // **Actually, every thread needs BLOCK_SIZE / blockDim.x shared memory.
    // Though, dynamic allocation would cause increased latency, and for now
    // total memory is enough for BLOCK_SIZE allocation by each thread.**
    double sq_dist_acc[BLOCK_SIZE];
    double kernel_tmp[BLOCK_SIZE];

    // Form nominator and denominator, by verticaly iterating over the chunks
    // of d_original.
    for (int stride_y = 0; stride_y < chunks_y; stride_y++) {

        denom_red[blockDim.x*threadIdx.y + cord] = 0.0;

        // Calculate the points available for computing. On last chunk, if
        // rows is not dividable by BLOCK_SIZE, available points will be less
        // than BLOCK_SIZE.
        int j_max;  // Max neighbours in current chunk available for calculation.
        if ((stride_y * BLOCK_SIZE + BLOCK_SIZE) <= rows) j_max = BLOCK_SIZE;
        else j_max = rows - stride_y * BLOCK_SIZE;

        // For each point in current block of points, initialize distance
        // accumulator to 0.0
        for (int j = cord; j < j_max; j += blockDim.x) {
            sq_dist_acc[j] = 0.0;
            kernel_tmp[j] = 0.0;
        }

        // Local row to be loaded by current thread (of d_original).
        int l_row = blockDim.y * threadIdx.x + threadIdx.y;
        int glob_row = stride_y * blockDim.y * blockDim.x + l_row;

    // =================== Calculate Square Distances =====================

        for (int stride_x = 0; stride_x < chunks_x; stride_x++) {

            // Calculate how many cords (columns) are available in current
            // chunk. On last chunk, less columns than SHARED_COLS will exist,
            // if number of columns is not dividable by SHARED_COLS.
            int k_max;
            if ((stride_x * SHARED_COLS + SHARED_COLS) <= cols) k_max = SHARED_COLS;
            else k_max = cols - stride_x * SHARED_COLS;
            int glob_col = stride_x * SHARED_COLS;

            // Local row may exceed rows if rows is not dividable by BLOCK_SIZE.
            if (l_row < j_max) {
                // Load local row into shared memory.
                for (int k = 0; k < k_max; k++) {
                    s_orig[l_row][k] = d_original[glob_row + lda*(glob_col+k)];
                }
            }

            // Make sure points in shared memory has been loaded by all threads
            // before starting computation.
            __syncthreads();

            if (i < rows) {
                // Compute the distances between current point and all original ones.
                for (int j = cord; j < j_max; j += blockDim.x) {

                    double sq_dist = 0;

                    // Calculate the square distance between current points pair.
                    for (int k = 0; k < k_max; k++) {

                        double d = d_shifted[i + (glob_col+k)*lda] - s_orig[j][k];
                        sq_dist += d * d;
                    }

                    // Only BLOCK_SIZE / blockDim.x cells are filled by each
                    // thread.
                    sq_dist_acc[j] += sq_dist;
                }
            }

            // Make sure calculations have been completed before loading new
            // chunk in shared memory.
            __syncthreads();
        }

    // ============ Calculate Kernels and Local Denominator Sum =============

        if (i < rows) {
            for (int j = threadIdx.x; j < j_max; j += blockDim.x) {
                double sq_dist = sq_dist_acc[j];

                // Ignore points on a distance greater than h.
                if (sq_dist < h*h) {
                    kernel_tmp[j] = exp(-sq_dist / (2*h*h));  // Calc kernel.

                    // Add to local denominator sum (reduce pending).
                    denom_red[blockDim.x*threadIdx.y + threadIdx.x] +=
                        kernel_tmp[j];
                }
            }

            // Make sure all values are ready for reduction.
            __syncthreads();

            // Denominator partial sums reduction.
            int half_threads = blockDim.x >> 1;  // Threads that can be utilized.

            while (half_threads > 0) {
                if (cord < half_threads && i < rows) {
                    int thread2 = cord + half_threads;

                    denom_red[blockDim.x*threadIdx.y + threadIdx.x] +=
                        denom_red[blockDim.x*threadIdx.y + thread2];
                }

                half_threads >>= 1;
                __syncthreads();
            }

            // Keep track of denominator inside each thread, so to utilize it
            // for parallel division with nominator's cords.
            if (i < rows) {
                denom += denom_red[blockDim.x*threadIdx.y];
            }
        }

    // ======== Calculate Local Nominator Sum and Final Reduction =========
    // ============= for both Nominator and Denominator. ==================

        // Repeat the process done for calulating the distances by utilizing
        // shared memory, for calculating nominator and denominator.
        for (int stride_x = 0; stride_x < chunks_x; stride_x++) {

            // Each thread initializes its own row into nom_red and denom_red,
            // where it will write its partial sum.
            for (int k = 0; k < SHARED_COLS; k++)
                nom_red[blockDim.x*threadIdx.y + cord][k] = 0.0;

            // Calculate how many cords (columns) are available in current
            // chunk. On last chunk, less columns than SHARED_COLS will exist,
            // if number of columns is not dividable by SHARED_COLS.
            int k_max;
            if ((stride_x * SHARED_COLS + SHARED_COLS) <= cols) k_max = SHARED_COLS;
            else k_max = cols - stride_x * SHARED_COLS;
            int glob_col = stride_x * SHARED_COLS;

            // Local row may exceed rows if rows is not dividable by BLOCK_SIZE.
            if (l_row < j_max) {
                // Load local row into shared memory.
                for (int k = 0; k < k_max; k++) {
                    s_orig[l_row][k] = d_original[glob_row + lda*(glob_col+k)];
                }
            }

            // Make sure points in shared memory has been loaded by all threads
            // before starting computation.
            __syncthreads();

            // Calculate nominator partial sums.
            if (i < rows) {
                for (int j = cord; j < j_max; j += blockDim.x) {
                    if (sq_dist_acc[j] < h*h) {
                        // Multiply each cord of each original point by the
                        // corresponding kernel.
                        for (int k = 0; k < k_max; k++) {
                            nom_red[blockDim.x*threadIdx.y + threadIdx.x][k] +=
                                s_orig[j][k] * kernel_tmp[j];
                        }
                    }
                }
            }

            // Make sure all values are ready for reduction.
            __syncthreads();

            // Nominator and denominator partial sums reduction.
            int half_threads = blockDim.x >> 1;  // Threads that can be utilized.

            while (half_threads > 0) {
                if (cord < half_threads && i < rows) {
                    int thread2 = cord + half_threads;

                    for (int k = 0; k < k_max; k++) {
                        nom_red[blockDim.x*threadIdx.y + cord][k] +=
                            nom_red[blockDim.x*threadIdx.y + thread2][k];
                    }
                }

                half_threads >>= 1;
                __syncthreads();
            }

            // Utilize as many as possible threads (<= k_max) to parallelize
            // column writing to d_n_work in global memory.
            if (i < rows) {
                for (int k = cord; k < k_max; k += blockDim.x) {
                    d_n_work[i + lda*(glob_col+k)] +=
                        nom_red[blockDim.x*threadIdx.y][k];
                }
            }

            // Make sure calculations have been completed before loading new
            // chunk in shared memory.
            __syncthreads();
        }
    }

    if (i < rows) {
        for (int k = cord; k < cols; k += blockDim.x) {
            // Calculate the new vector of actual shifted point.
            double new_cord = d_n_work[i + k*lda] / denom;

            // Calculate mean shift vector for current point
            d_m[i + k*lda] = new_cord - d_shifted[i + k*lda];

            // Finally, update the previously shifted vector with the new one.
            d_shifted[i + k*lda] = new_cord;
        }
    }
}
