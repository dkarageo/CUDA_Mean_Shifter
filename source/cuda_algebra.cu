#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <cusolverDn.h>

extern "C" {
#include "cuda_algebra.h"
}

extern "C"
double cuda_norm(double *d_A, int m, int n, int lda)
{
    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;

    // Allocate S array where to retrieve singular values.
    double *S = (double *) malloc(sizeof(double) * n);

    double *d_S = NULL;
    int *devInfo = NULL;
    double *d_work = NULL;
    double *d_rwork = NULL;

    int lwork = 0;

    // Create cusolverDn handle.
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    // Create a copy of given matrix, since it is going to be destroyed after
    // calculating SVD.
    double *d_A_cp = NULL;
    cudaStat1 = cudaMalloc((void **) &d_A_cp, sizeof(double)*lda*n);
    assert(cudaSuccess == cudaStat1);
    cudaStat1 =
        cudaMemcpy(d_A_cp, d_A, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice);
    assert(cudaSuccess == cudaStat1);
    d_A = d_A_cp;

    // Allocate helper matrixes and utils.
    cudaStat1 = cudaMalloc((void **) &d_S, sizeof(double)*n);
    cudaStat2 = cudaMalloc((void **) &devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    // Query working space of SVD.
    cusolver_status = cusolverDnDgesvd_bufferSize(
            cusolverH,
            m,
            n,
            &lwork);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cudaStat1 = cudaMalloc((void **) &d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

    // Compute SVD.
    signed char jobu = 'N';  // all m columns of U
    signed char jobvt = 'N';  // all n columns of VT
    cusolver_status = cusolverDnDgesvd(
            cusolverH,
            jobu,
            jobvt,
            m,
            n,
            d_A,
            lda,
            d_S,
            NULL,
            lda,  // ldu
            NULL,
            lda,  // ldvt
            d_work,
            lwork,
            d_rwork,
            devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(S, d_S, sizeof(double)*n, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    // Calculate frobenius norm.
    double frob = 0.0;
    for (int i = 0; i < n; i++) frob += powf(S[i], 2.0);
    frob = sqrt(frob);

    if (cusolverH) cusolverDnDestroy(cusolverH);

    // Release resources.
    free(S);
    if (d_S) cudaFree(d_S);
    if (devInfo) cudaFree(devInfo);
    if (d_work) cudaFree(d_work);
    if (d_rwork) cudaFree(d_rwork);
    if (d_A_cp) cudaFree(d_A_cp);

    return frob;
}
