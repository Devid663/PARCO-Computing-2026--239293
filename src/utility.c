#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utily.h"

//csr requires a row grouping, so it converts the coo grouping
int create_sparse_csr_from_coo(int nrows, int ncols, int nnz, int *coo_rows, int *coo_cols, double *coo_vals, sparse_csr *csr){
    csr->nrows = nrows;   //store matrix dimensions
    csr->ncols = ncols;
    csr->nnz   = nnz;

    csr->row_ptr = calloc(nrows + 1, sizeof(size_t)); //allocate CSR arrays
    csr->col_ind = malloc(nnz * sizeof(size_t));
    csr->val     = malloc(nnz * sizeof(double));

    if ( !csr->row_ptr || !csr->col_ind || !csr->val ) { //check failures
        return EXIT_FAILURE; 
    }

    int *row_count = calloc(nrows, sizeof(int));   //temporary array to count nnz per row

    for (int k = 0; k < nnz; k++) {   //counts each non-zero row element
        row_count[coo_rows[k]]++;
    }

    csr->row_ptr[0] = 0;     //build the row_ptr array           
    for (int i = 0; i < nrows; i++) {
        csr->row_ptr[i + 1] = csr->row_ptr[i] + row_count[i];
    }

    for (int i = 0; i < nrows; i++) {  //reset row_count
        row_count[i] = 0;
    }

    for (int k = 0; k < nnz; k++) {    //fill col ind e val in CSR
        int row = coo_rows[k];
        int dest = csr->row_ptr[row] + row_count[row];
        csr->col_ind[dest] = coo_cols[k];
        csr->val[dest] = coo_vals[k];
        row_count[row]++;
    }

    free(row_count);
    return EXIT_SUCCESS;
}

//prints the matrix values
void print_sparse_csr(const sparse_csr* csr){
    printf("row\tcol\tval\n");
    printf("---\n");
    for (size_t i = 0; i < csr->nrows; ++i) {
        size_t nz_start = csr->row_ptr[i];
        size_t nz_end = csr->row_ptr[i + 1];
        for(size_t nz_id = nz_start; nz_id < nz_end; ++nz_id){
            size_t j = csr->col_ind[nz_id];
            double val = csr->val[nz_id];
            printf("%d\t%d\t%02.2f\n", i, j, val);
        }
    }
}

//multiplies eache nnz with each v element
void matrix_vector_sparse_csr(const sparse_csr* csr, const double* v, double* res){
    for (size_t i = 0; i < csr->nrows; ++i) {
        res[i] = 0.0;
        size_t nz_start = csr->row_ptr[i];
        size_t nz_end = csr->row_ptr[i + 1];
        for(size_t nz_id = nz_start; nz_id < nz_end; ++nz_id){
            size_t j = csr->col_ind[nz_id];
            double val = csr->val[nz_id];
            res[i] = res[i] + val * v[j];
        }
    }
}

//deallocates the arrays of csr struct
int free_sparse_csr(sparse_csr* csr) {
    free(csr->row_ptr);
    free(csr->col_ind);
    free(csr->val);

    return EXIT_SUCCESS;
}

//------PARALLEL VERSIONS------

//parallel mupliplication in fixed scheduling
void matrix_vector_sparse_csr_omp_static(const sparse_csr* csr, const double* v, double* res) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < csr->nrows; ++i) {
        double sum = 0.0;
        for (size_t j = csr->row_ptr[i]; j < csr->row_ptr[i + 1]; ++j)
            sum += csr->val[j] * v[csr->col_ind[j]];
        res[i] = sum;
    }
}

//parallel mupliplication in dynamic scheduling
void matrix_vector_sparse_csr_omp_dynamic(const sparse_csr* csr, const double* v, double* res) {
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < csr->nrows; ++i) {
        double sum = 0.0;
        for (size_t j = csr->row_ptr[i]; j < csr->row_ptr[i + 1]; ++j)
            sum += csr->val[j] * v[csr->col_ind[j]];
        res[i] = sum;
    }
}

//parallel mupliplication in dec. ordered scheduling
void matrix_vector_sparse_csr_omp_guided(const sparse_csr* csr, const double* v, double* res) {
#pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < csr->nrows; ++i) {
        double sum = 0.0;
        for (size_t j = csr->row_ptr[i]; j < csr->row_ptr[i + 1]; ++j)
            sum += csr->val[j] * v[csr->col_ind[j]];
        res[i] = sum;
    }
}

//checks the result of the sequential calulation and parallel version
int check_results(const double *res_seq, const double *res_par, size_t n) {
    const double eps = 1e-8;
    for (size_t i = 0; i < n; i++) {
        if (fabs(res_seq[i] - res_par[i]) > eps) {
            printf("Error: mismatch at index %zu : seq = %f, par = %f\n", i, res_seq[i], res_par[i]);
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}


