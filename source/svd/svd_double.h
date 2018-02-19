/**
 * svd_double.h
 *
 * Created by Dimitrios Karageorgiou,
 * for course "Parallel And Distributed Systems".
 * Electrical and Computers Engineering Department, AuTh, GR - 2017-201
 *
 * A header file exposing a function for singular value decompositon of
 * doubles matrix.
 */

#ifndef __svd_double_h__
#define __svd_double_h__


int dsvd(double **a, int m, int n, double *w, double **v);


#endif
