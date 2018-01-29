#include "matrix.h"


matrix_t *mean_shift(matrix_t *points, double h, double e);
void shift_point(matrix_t *shifted, matrix_t *m, int i, matrix_t *original,
                 double h);
