#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "mmio.h"
#include "utility.h"

int main(int argc, char** argv) {
    srand(time(NULL));

    FILE *f;
    MM_typecode matcode;
    int ret_code;
    int nrows, ncols, nz; // matrix.mtx dimensions: rows, columns, no zero
    mm_initialize_typecode(&matcode);

    if (argc < 2) {
        printf("Usage: %s matrix_file.mtx\n", argv[0]);
        return EXIT_FAILURE;
    }

    //opens the matrix.mtx file
    if ((f = fopen(argv[1], "r")) == NULL) {
        perror("Cannot open file");
        return EXIT_FAILURE;
    }
    printf("Opened file successfully!\n");


    //reading the headers of the matrix.mtx
    if (mm_read_banner(f, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        return EXIT_FAILURE;
    }

    int is_pattern = mm_is_pattern(matcode);  //matrix pattern type
    int is_real    = mm_is_real(matcode);
    int is_integer = mm_is_integer(matcode);
    int is_complex = mm_is_complex(matcode);

    //accepts only real/integer/pattern type
    if (!mm_is_coordinate(matcode)) {
    fprintf(stderr, "Error: this code only supports COO (coordinate) format.\n");
    return EXIT_FAILURE;
    }

    if (is_complex) {
        fprintf(stderr, "Error: complex matrices are not supported.\n");
        return EXIT_FAILURE;
    }

    if (!mm_is_general(matcode)) {
        fprintf(stderr, "Error: symmetric/skew/hermitian matrices not supported.\n");
        return EXIT_FAILURE;
    }

    if ((ret_code = mm_read_mtx_crd_size(f, &nrows, &ncols, &nz)) != 0)  //read matrix dimensions
        return ret_code;

    printf("Matrix dimensions: %d x %d with %d non-zero entries\n", nrows, ncols, nz);

    int *I = malloc(nz * sizeof(int));  //allocation of coo rapresentation
    int *J = malloc(nz * sizeof(int));
    double *val = malloc(nz * sizeof(double));

    for (int k = 0; k < nz; k++) {  //read the coo entries from the file
        int row, col;
        double value = 1.0;  
    
        if (is_pattern) {  //if is in pattern type values are implicitly 1
            if (fscanf(f, "%d %d", &row, &col) != 2) {
                fprintf(stderr, "Error reading pattern entry %d\n", k);
                return EXIT_FAILURE;
            }
        } else {
            if (fscanf(f, "%d %d %lf", &row, &col, &value) != 3) {
                fprintf(stderr, "Error reading valued entry %d\n", k);
                return EXIT_FAILURE;
            }
        }
        
        I[k]   = row - 1;  //converts the Matrix indices
        J[k]   = col - 1;
        val[k] = value;
    }

    fclose(f);

    sparse_csr csr;
    create_sparse_csr_from_coo(nrows, ncols, nz, I, J, val, &csr);
    //print_sparse_csr(&csr);

    double *v = malloc(ncols * sizeof(double));  //vector
    double *res_seq = malloc(nrows * sizeof(double)); //sequential result
    double *res_static = malloc(nrows * sizeof(double)); //parallel static resutl
    double *res_dynamic = malloc(nrows * sizeof(double));  //parallel dynamic resutl
    double *res_guided = malloc(nrows * sizeof(double));  //parallel guided result

    for (int i = 0; i < ncols; i++) {  //random vector generation
        v[i] = (double) rand() / RAND_MAX;
        //    printf("v[%d] = %g\n", i, v[i]);
    }

    //||||||||||||||||||CHECK RESULTS|||||||||||||||||||||||||||||||
    //SEQUENTIAL
    matrix_vector_sparse_csr(&csr, v, res_seq);

    //PARALLEL STATIC
    matrix_vector_sparse_csr_omp_static(&csr, v, res_static);
    //CHECKING THE RESULTS
    if (check_results(res_seq, res_static, nrows) != EXIT_SUCCESS) {
        printf("Static version incorrect. Aborting.\n");
        return EXIT_FAILURE;
    }

    //PARALLEL DYNAMIC
    matrix_vector_sparse_csr_omp_dynamic(&csr, v, res_dynamic);
    //CHECKING THE RESULTS
    if (check_results(res_seq, res_dynamic, nrows) != EXIT_SUCCESS) {
        printf("Dynamic version incorrect. Aborting.\n");
        return EXIT_FAILURE;
    }

    //PARALLEL GUIDED
    matrix_vector_sparse_csr_omp_guided(&csr, v, res_guided);
    //CHECKING THE RESULTS
    if (check_results(res_seq, res_guided, nrows) != EXIT_SUCCESS) {
        printf("Guided version incorrect. Aborting.\n");
        return EXIT_FAILURE;
    }

    //|||||||||||||||||||BENCHMARKS|||||||||||||||||||||||||||||||
    double start_time;
    double end_time;

    double times_seq[10];
    //EXECUTING 10 TIMES THE SEQUENTAIL
    for (int j = 0; j < 10; j++) {
        start_time = omp_get_wtime();

        matrix_vector_sparse_csr(&csr, v, res_seq);

        end_time = omp_get_wtime();
        times_seq[j] = (end_time - start_time) * 1000.0;
        printf("seq %d %d %.6f\n", 1, j, times_seq[j]);
        /*
        printf("\nResult (A * v) in [sequential mode]:\n");
        for (int i = 0; i < M; i++)
            printf("%02.2f\n", res_seq[i]);
        */
    }

    //||||||||||||||||PARALLEL OPERATIONS||||||||||||||||
    double times_sta[10];
    //EXECUTING 1 TIME FOR CACHE WARMUP
    matrix_vector_sparse_csr_omp_static(&csr, v, res_static);

    //EXECUTING 10 TIMES
    for (int j = 0; j < 10; j++) {
        start_time = omp_get_wtime();

        matrix_vector_sparse_csr_omp_static(&csr, v, res_static);

        end_time = omp_get_wtime();
        times_sta[j] = (end_time - start_time) * 1000.0;
        printf("static %d %d %.6f\n", omp_get_max_threads(), j, times_sta[j]);

        /*
        printf("\nResult (A * v) in [parallel mode (static omp)]:\n");
        for (int i = 0; i < M; i++)
            printf("%02.2f\n", res_static[i]);
        */
    }

    //EXECUTING 1 TIME FOR CACHE WARMUP
    double times_dyn[10];
    matrix_vector_sparse_csr_omp_dynamic(&csr, v, res_dynamic);

    //EXECUTING 10 TIMES
    for (int j = 0; j < 10; j++) {
        start_time = omp_get_wtime();

        matrix_vector_sparse_csr_omp_dynamic(&csr, v, res_dynamic);

        end_time = omp_get_wtime();
        times_dyn[j] = (end_time - start_time) * 1000.0;
        printf("dynamic %d %d %.6f\n", omp_get_max_threads(), j, times_dyn[j]);
        /*
        printf("\nResult (A * v) in [parallel mode (dynamic omp)]:\n");
        for (int i = 0; i < M; i++)
            printf("%02.2f\n", res_dynamic[i]);
        */
    }

    //EXECUTING 1 TIME FOR CACHE WARMUP
    double times_guided[10];
    matrix_vector_sparse_csr_omp_guided(&csr, v, res_guided);

    //EXECUTING THE PRODUCT 10 TIMES BENCHMARKING
    for (int j = 0; j < 10; j++) {
        start_time = omp_get_wtime();

        matrix_vector_sparse_csr_omp_guided(&csr, v, res_guided);

        end_time = omp_get_wtime();
        times_guided[j] = (end_time - start_time) * 1000.0;
        printf("guided %d %d %.6f\n", omp_get_max_threads(), j, times_guided[j]);
        /*
        printf("\nResult (A * v) in [parallel mode (dynamic omp)]:\n");
        for (int i = 0; i < M; i++)
            printf("%02.2f\n", res_guided[i]);
        */
    }

    //deallocation
    free_sparse_csr(&csr);
    free(I);
    free(J);
    free(val);
    free(v);
    free(res_seq);
    free(res_static);
    free(res_dynamic);
    free(res_guided);

    return 0;
}