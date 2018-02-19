/**
 * matrix.h
 *
 * Created by Dimitrios Karageorgiou,
 *  for course "Parallel And Distributed Systems".
 *  Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
 *
 * matrix.h defines routines that may be used for creating and managing
 * matrices of doubles.
 *
 * Version: 0.2
 *
 * Types defined in knn.h:
 *  -matrix_t (opaque)
 *
 * Macros defined in knn.h:
 *	-matrix_get_cols(matrix)
 *	-matrix_get_rows(matrix)
 *	-matrix_get_cell(matrix, row, col)
 *	-matrix_get_chunk_offset(matrix)
 *	-matrix_set_cell(matrix, row, col, value)
 *
 * Functions defined in knn.h:
 *	-matrix_t *matrix_create(int32_t rows, int32_t cols)
 *	-void matrix_destroy(matrix_t *matrix)
 *	-matrix_t *matrix_load_in_chunks(const char *filename,
 *	   								 int32_t chunks_num,
 * 									 int32_t req_chunk)
 *	-char *matrix_serialize(matrix_t *matrix, size_t *bytec)
 *	-matrix_t *matrix_deserialize(char *bytes, size_t bytec)
 *  -double **matrix_to_2d_array(matrix_t *matrix)
 *  -matrix_t *matrix_create_copy(matrix_t *matrix)
 *  -matrix_t *matrix_fill(matrix_t *matrix, double value)
 *  -matrix_t *matrix_set_row(matrix_t *matrix, int32_t row,
 *                         matrix_t *new_matrix, int32_t new_row)
 *  -matrix_t *matrix_row_num_mul(matrix_t *matrix, int32_t row, double num,
 *                             matrix_t *out, int32_t out_row)
 *  -matrix_t *matrix_row_num_div(matrix_t *matrix, int32_t row, double num,
 *                             matrix_t *out, int32_t out_row)
 *  -matrix_t *matrix_row_add(matrix_t *m1, int32_t r1, matrix_t *m2, int32_t r2,
 *                         matrix_t *out, int32_t out_row)
 *  -matrix_t *matrix_row_sub(matrix_t *m1, int32_t r1, matrix_t *m2, int32_t r2,
 *                         matrix_t *out, int32_t out_row)
 */

#ifndef __matrix_h__
#define __matrix_h__


#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
	double **data;             // Actual data of the matrix.
	int32_t rows;              // Rows counter of the matrix.
	int32_t cols;              // Columns counter of the matrix.
    int32_t chunk_offset;      // Offset of this matrix, in its container
                               // array, if it belongs to any.
} matrix_t;


/**
 * Returns number of columns of the given matrix.
 */
#define matrix_get_cols(matrix) matrix->cols

/**
 * Returns number of rows of the given matrix.
 */
#define matrix_get_rows(matrix) matrix->rows

/**
 * Returns the value of matrix cell.
 */
#define matrix_get_cell(matrix, row, col) matrix->data[row][col]

/**
 * Returns the offset of the first row of current matrix chunk
 * from the beggining of the complete matrix.
 */
#define matrix_get_chunk_offset(matrix) matrix->chunk_offset

/**
 * Sets the value of a matrix cell to the given one.
 */
#define matrix_set_cell(matrix, row, col, value) matrix->data[row][col] = value

/**
 * Creates a new empty matrix object.
 *
 * Elements are not initialized, and their initial value is undefined.
 *
 * Parameters:
 *	-rows: Number of rows the new matrix will contain.
 *	-cols: Number of cols the new matrix will contain.
 *
 * On successful creation returns the matrix object. On failure, returns
 * NULL.
 */
matrix_t *matrix_create(int32_t rows, int32_t cols);

/**
 * Destroys a matrix object and releases its resources.
 *
 * Parameters:
 *	-matrix: The matrix to destroy.
 */
void matrix_destroy(matrix_t *matrix);

/**
 * Loads a chunk of a matrix object stored to filesystem.
 *
 * Matrix is separated in chunks only by rows. Thus, each chunk will contain
 * a portions of the total rows, though each one with all its columns.
 * Chunks should be as equal in size as possible. When differ,
 * the smallest from the biggest chunk won't differ by more than 1 rows.
 * Chunks are 0-indexed and their indexes range into [0, chunks_num-1].
 *
 * In order to load the complete matrix, chunks_num can
 * be set to 1 and req_chunk to 0.
 *
 * When loading a chunk, the offset of its first row from the beggining of the
 * complete matrix can be queried by using matrix_get_chunk_offset() function.
 *
 * Parameters:
 *	-filename: A path to a file that stores a matrix object.
 * 	-chunks_num: The total number of chunks the matrix should be divided into.
 *	-req_chunk: The index of chunk to be loaded. It ranges into [0, chunks_num-1].
 *
 * Returns:
 *	On successful loading, the matrix object corresponding to requested chunk.
 *	On failure, returns NULL.
 */
matrix_t *matrix_load_in_chunks(const char *filename,
								int32_t chunks_num,
								int32_t req_chunk);

/**
 * Serializes the given matrix object.
 *
 * Parameters:
 *	-matrix: The matrix to serialize.
 *	-bytec: A reference to the location to write the size in bytes of serialized
 *			object.
 *
 * Returns:
 *	On success, a reference to the serialized object. On failure, it returns
 *	NULL.
 */
char *matrix_serialize(matrix_t *matrix, size_t *bytec);

/**
 * Inflates a matrix object, out of its serial representaton.
 *
 * Parameters:
 *	-bytes: A reference to the serial representation of the matrix.
 *	-bytec: The size of the serial representation.
 *
 * Returns:
 *	On success returns a matrix object. On failure returns NULL.
 */
matrix_t *matrix_deserialize(char *bytes, size_t bytec);

/**
 * Returns a reference to the current matrix, as a pointer to pointer
 * representation.
 *
 * Returned reference is guaranteed to be in row-major order.
 *
 * No new matrix is allocated. The returned reference is guaranteed to be
 * just a handler to the given matrix. This also does NOT imply that returned
 * handler is the way each matrix is implemented. The only thing guaranteed is
 * that returned handler and all allocated memory for it, will be deallocated
 * upon the destruction of the matrix object bound with.
 *
 * Parameters:
 *  -matrix : A matrix object whose 2D array representation is requested.
 *
 * Returns:
 *  -A reference to a 2D array looking handler for given matrix.
 */
double **matrix_to_2d_array(matrix_t *matrix);

/**
 * Copies the given matrix into a newly allocated one.
 *
 * Parameters:
 *  -matrix : Matrix to copy.
 *
 * Returns:
 *  A reference to a new matrix object containing the contents of the given one.
 */
matrix_t *matrix_create_copy(matrix_t *matrix);

/**
 * Fills all the cells of a given matrix with the given value.
 *
 * Parameters:
 *  -matrix : The matrix to be filled.
 *  -value : The value to be placed on each cell of the given matrix.
 *
 * Returns:
 *  -A reference to the given matrix.
 */
matrix_t *matrix_fill(matrix_t *matrix, double value);

/**
 * Copies a row from one matrix to another.
 *
 * Parameters:
 *  -matrix : A matrix into which the row will be copied.
 *  -row : The index of row where new row will be copied.
 *  -new_matrix : The source of the row to be copied into matrix.
 *  -new_row : The index of row in new_matrix to be copied to the given index
 *          in matrix.
 *
 * Returns:
 *  The value of matrix.
 */
matrix_t *matrix_set_row(matrix_t *matrix, int32_t row,
                         matrix_t *new_matrix, int32_t new_row);

/**
 * Multiplies a row by a number.
 *
 * Parameters:
 *  -matrix : The matrix containing the row to be the multiplicand.
 *  -row : Index of multiplicand row in matrix.
 *  -num : The multiplier.
 *  -out : A matrix into which the result will be written. If it is NULL, a new
 *          row-matrix will be allocated for the result. If (out_row == matrix),
 *          the result is written in the original matrix.
 *  -out_row : (Valid only if out != NULL) The index of row in out matrix,
 *          where the result of multiplication will be written.
 *
 * Returns:
 *  A reference to the matrix containing the result.
 */
matrix_t *matrix_row_num_mul(matrix_t *matrix, int32_t row, double num,
                             matrix_t *out, int32_t out_row);

/**
 * Divides a row by a number.
 *
 * Parameters:
 *  -matrix : The matrix containing the row to be divided.
 *  -row : Index of row in matrix to be divided.
 *  -num : The divisor.
 *  -out : A matrix into which the result will be written. If it is NULL, a new
 *          row-matrix will be allocated for the result. If (out_row == matrix),
 *          the result is written in the original matrix.
 *  -out_row : (Valid only if out != NULL) The index of row in out matrix,
 *          where the result of division will be written.
 *
 * Returns:
 *  A reference to the matrix containing the result.
 */
matrix_t *matrix_row_num_div(matrix_t *matrix, int32_t row, double num,
                             matrix_t *out, int32_t out_row);

/**
 * Adds two rows.
 *
 * Parameters:
 *  -m1 : Matrix containing the first row.
 *  -r1 : Index of first row in m1.
 *  -m2 : Matrix containing the second row.
 *  -r2 : index of second row in m2.
 *  -out : A matrix into which the result will be written. If it is NULL, a new
 *          row-matrix will be allocated for the result. If (out_row == matrix),
 *          the result is written in the original matrix.
 *  -out_row : (Valid only if out != NULL) The index of row in out matrix,
 *          where the result of addition will be written.
 *
 * Returns:
 *  A reference to the matrix containing the result.
 */
matrix_t *matrix_row_add(matrix_t *m1, int32_t r1, matrix_t *m2, int32_t r2,
                         matrix_t *out, int32_t out_row);

/**
 * Subtracts one row from another, (i.e. row1 - row2).
 *
 * Parameters:
 *  -m1 : Matrix containing the subtrahend row.
 *  -r1 : Index of subtrahend row in m1.
 *  -m2 : Matrix containing the subtracter row.
 *  -r2 : index of subtracter row in m2.
 *  -out : A matrix into which the result will be written. If it is NULL, a new
 *          row-matrix will be allocated for the result. If (out_row == matrix),
 *          the result is written in the original matrix.
 *  -out_row : (Valid only if out != NULL) The index of row in out matrix,
 *          where the result of subtraction will be written.
 *
 * Returns:
 *  A reference to the matrix containing the result.
 */
matrix_t *matrix_row_sub(matrix_t *m1, int32_t r1, matrix_t *m2, int32_t r2,
                         matrix_t *out, int32_t out_row);

#ifdef __cplusplus
}
#endif
#endif
