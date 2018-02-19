/**
 * cuda_algebra.h
 *
 * Created by Dimitrios Karageorgiou,
 *  for course "Parallel And Distributed Systems".
 *  Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
 *
 * This header provides routines for common linear algebra calculations,
 * that utilize cuda enabled GPUs.
 *
 * Functions defined in cuda_algebra.h:
 *  -double cuda_norm(double *d_A, int m, int n, int lda)
 */

#ifndef __cuda_algebra_h__
#define __cuda_algebra_h__


/**
 * Calculates the Frobenius Norm of a matrix, stored in device memory.
 *
 * Parameters:
 *  -d_A : An (lda*n) doubles' matrix stored in device memory, in column-major
 *          order.
 *  -m : The number of rows of d_A.
 *  -n : The number of columns of d_A.
 *  -lda : The leading dimension of d_A. This allows the usage of matrices
 *          that in their linear representation contain padding between their
 *          successive columns. If now padding exists, then this argument
 *          should be set to the value given in m (number of rows).
 *
 * Returns:
 *  The frobenius norm of the given matrix.
 */
double cuda_norm(double *d_A, int m, int n, int lda);

#endif
