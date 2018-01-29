/**
 * matrix.c
 *
 * Created by Dimitrios Karageorgiou,
 *  for course "Parallel And Distributed Systems".
 *  Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
 *
 * matrix.c provides an implementation for routines defined in matrix.h.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "matrix.h"

matrix_t *matrix_create(int32_t rows, int32_t cols)
{
	// Allocate a new matrix_t object.
	matrix_t *matrix = (matrix_t *) malloc(sizeof(matrix_t));
	if (matrix == NULL) return NULL;

	matrix->rows = rows;
	matrix->cols = cols;

	// Allocate memory for rows * cols doubles.
	matrix->data = (double **) malloc(sizeof(double *) * rows);
	if (matrix->data == NULL) return NULL;  // -- Memory leak, cleanup pending. --

	for (int32_t i = 0; i < rows; i++) {
		matrix->data[i] = (double *) malloc(sizeof(double) * cols);
		if (matrix->data[i] == NULL) return NULL;  // -- Memory leak, cleanup pending. --
	}

    // A newly created matrix is always considered to be the first in its
    // container array.
    matrix->chunk_offset = 0;

	return matrix;
}

void matrix_destroy(matrix_t *matrix)
{
    for (int32_t i = 0; i < matrix->rows; i++) {
		free(matrix->data[i]);
	}
    free(matrix->data);
	free(matrix);
}

matrix_t *matrix_load_in_chunks(const char *filename,
								int32_t chunks_num,
								int32_t req_chunk)
{
    FILE *f = fopen(filename, "r");
    if (f == NULL) {
        printf("ERROR: matrix_load_in_chunks : Failed to open %s\n", filename);
        return NULL;
    }

	int32_t total_rows;  // Total rows contained in file.
	int32_t rows;  // Rows contained in requested chunk.
	int32_t cols;  // Columns of the matrix.

	// Read total rows counter.
	int rc = fread(&total_rows, sizeof(int32_t), 1, f);
	if (rc != 1) {
		printf("ERROR: Failed to read rows number.\n");
		exit(-1);
	}

	// Read columns counter.
	rc = fread(&cols, sizeof(int32_t), 1, f);
	if (rc != 1) {
		printf("ERROR: Failed to read cols number.\n");
		exit(-1);
	}

	// Break the file into given number of chunks and set file
	// position to the beggining of the requested chunk.
	rows = total_rows / chunks_num;
	int remaining = total_rows % chunks_num;
	long int offset;
	if (req_chunk < remaining) {
		rows++;
		offset = req_chunk * rows;
	} else {
		offset = ((rows + 1) * remaining) + (rows * (req_chunk - remaining));
	}

	rc = fseek(f, sizeof(double) * offset * cols, SEEK_CUR);

	matrix_t *matrix = matrix_create(rows, cols);
    matrix->chunk_offset = offset;  // Set the offset of the matrix object
                                    // to the number of rows from the beggining
                                    // of the file, till the first row to
                                    // be included in this matrix.

	// Read data from file, row by row.
	for(int32_t i = 0; i < rows; i++) {
		rc = fread(matrix->data[i], sizeof(double), cols, f);
		if (rc != cols) {
			printf("Failed in reading col %d. \n", i+1);
			exit(-1);
		}
	}

	fclose(f);

	return matrix;
}

char *matrix_serialize(matrix_t *matrix, size_t *bytec)
{
    int32_t rows = matrix_get_rows(matrix);
    int32_t cols = matrix_get_cols(matrix);
    int32_t offset = matrix_get_chunk_offset(matrix);

    // Allocate space for 3 ints (rows and columns counter, offset) and all cells.
    *(bytec) = sizeof(int32_t) * 3 + sizeof(double) * rows * cols;

    char *serialized = (char *) malloc (sizeof(char) * (*bytec));
    if (!serialized) {
        printf("ERROR: matrix_serialize : Failed to allocate memory.\n");
        return NULL;
    }

    // Write matrix to its serialized form.
    char *buffer = serialized;

    memcpy(buffer, &rows, sizeof(int32_t)); buffer += sizeof(int32_t);
    memcpy(buffer, &cols, sizeof(int32_t)); buffer += sizeof(int32_t);
    memcpy(buffer, &offset, sizeof(int32_t)); buffer += sizeof(int32_t);

    for (int32_t i = 0; i < rows; i++) {
        memcpy(buffer, matrix->data[i], sizeof(double) * cols);
        buffer += sizeof(double) * cols;
    }

    return serialized;
}

matrix_t *matrix_deserialize(char *bytes, size_t bytec)
{
    char *buffer = bytes;

    int32_t rows;
    int32_t cols;
    int32_t offset;

    memcpy(&rows, buffer, sizeof(int32_t)); buffer += sizeof(int32_t);
    memcpy(&cols, buffer, sizeof(int32_t)); buffer += sizeof(int32_t);
    memcpy(&offset, buffer, sizeof(int32_t)); buffer += sizeof(int32_t);

    if (bytec != sizeof(int32_t) * 3 + sizeof(double) * rows * cols) {
        printf("ERROR: matrix_deserialize : Given and actual size not matching.\n");
        return NULL;
    }

    matrix_t *matrix = matrix_create(rows, cols);
    matrix->chunk_offset = offset;

    for (int32_t i = 0; i < rows; i++) {
        memcpy(matrix->data[i], buffer, sizeof(double) * cols);
        buffer += sizeof(double) * cols;
    }

    return matrix;
}

double **matrix_to_2d_array(matrix_t *matrix) { return matrix->data; }

matrix_t *matrix_create_copy(matrix_t *matrix)
{
    int32_t rows = matrix_get_rows(matrix);
    int32_t cols = matrix_get_cols(matrix);

    matrix_t *newm = matrix_create(rows, cols);

    newm->chunk_offset = matrix->chunk_offset;

    for (int i = 0; i < rows; i++) {
        memcpy(newm->data[i], matrix->data[i], cols * sizeof(double));
    }

    return newm;
}

matrix_t *matrix_fill(matrix_t *matrix, double value)
{
    int32_t rows = matrix_get_rows(matrix);
    int32_t cols = matrix_get_cols(matrix);

    double *initializer = (double *) malloc(sizeof(double) * cols);
    for (int i = 0; i < cols; i++) initializer[i] = value;

    for (int i = 0; i < rows; i++) {
        memcpy(matrix->data[i], initializer, sizeof(double) * cols);
    }

    free(initializer);
    return matrix;
}

matrix_t *matrix_set_row(matrix_t *matrix, int32_t row,
                         matrix_t *new_matrix, int32_t new_row)
{
    memcpy(matrix->data[row], new_matrix->data[new_row],
           matrix_get_cols(matrix) * sizeof(double));

    return matrix;
}

matrix_t *matrix_row_num_mul(matrix_t *matrix, int32_t row, double num,
                             matrix_t *out, int32_t out_row)
{
    if (out == NULL) {
        out = matrix_create(1, matrix_get_cols(matrix));
        out_row = 0;
    }

    for (int j = 0; j < matrix_get_cols(matrix); j++) {
        matrix_set_cell(out, out_row, j, matrix_get_cell(matrix, row, j) * num);
    }

    return out;
}

matrix_t *matrix_row_num_div(matrix_t *matrix, int32_t row, double num,
                             matrix_t *out, int32_t out_row)
{
    if (out == NULL) {
        out = matrix_create(1, matrix_get_cols(matrix));
        out_row = 0;
    }

    for (int j = 0; j < matrix_get_cols(matrix); j++) {
        matrix_set_cell(out, out_row, j, matrix_get_cell(matrix, row, j) / num);
    }

    return out;
}

matrix_t *matrix_row_add(matrix_t *m1, int32_t r1, matrix_t *m2, int32_t r2,
                         matrix_t *out, int32_t out_row)
{
    if (out == NULL) {
        out = matrix_create(1, matrix_get_cols(m1));
        out_row = 0;
    }

    for (int j = 0; j < matrix_get_cols(m1); j++) {
        matrix_set_cell(
            out, out_row, j,
            matrix_get_cell(m1, r1, j) + matrix_get_cell(m2, r2, j)
        );
    }

    return out;
}

matrix_t *matrix_row_sub(matrix_t *m1, int32_t r1, matrix_t *m2, int32_t r2,
                         matrix_t *out, int32_t out_row)
{
    if (out == NULL) {
        out = matrix_create(1, matrix_get_cols(m1));
        out_row = 0;
    }

    for (int j = 0; j < matrix_get_cols(m1); j++) {
        matrix_set_cell(
            out, out_row, j,
            matrix_get_cell(m1, r1, j) - matrix_get_cell(m2, r2, j)
        );
    }

    return out;
}
