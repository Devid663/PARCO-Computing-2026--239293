#ifndef UTILITY_H
#define UTILITY_H

#include <stdio.h>

//struct for the matrix in csr format
typedef struct sparse_CSR {
    size_t nrows; //number of rows
    size_t ncols; //number of columns
    size_t nnz; //number of no zero entries
    size_t* row_ptr; //array of row pointers
    size_t* col_ind; //array of column indices
    double* val; //array of values
} sparse_csr;

//--------------------------------------SEQUENTIAL FUNCTIONS--------------------------------------
//function for converting into csr sparse matrix.mtx
int create_sparse_csr_from_coo(int nrows, int ncols, int nnz, int *coo_rows, int *coo_cols, double *coo_vals, sparse_csr *csr);
//function for printing the matrix.mtx
void print_sparse_csr(const sparse_csr* csr);
//function for performing the sequential operation
void matrix_vector_sparse_csr(const sparse_csr* csr, const double* v, double* res);
int free_sparse_csr(sparse_csr* csr);
//function for checking the result between sequential and parallel
int check_results(const double *res_seq, const double *res_par, size_t n);

//--------------------------------------PARALLEL FUNCTIONS--------------------------------------
void matrix_vector_sparse_csr_omp_static(const sparse_csr* csr, const double* v, double* res);
void matrix_vector_sparse_csr_omp_dynamic(const sparse_csr* csr, const double* v, double* res);
void matrix_vector_sparse_csr_omp_guided(const sparse_csr* csr, const double* v, double* res);

#endif