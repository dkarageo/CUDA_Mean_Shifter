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
 */
double cuda_norm(double *d_A, int m, int n, int lda);

#endif
