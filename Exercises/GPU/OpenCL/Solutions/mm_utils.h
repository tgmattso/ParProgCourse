//
// include file to support our matrix utility functions
//
// The only parameter you may want to change sometimes is TYPE
//
//  We use float since all OpenCL platforms must support float.
//  If you realy want to support double, however, you can do so
//  by changing TYPE in this file and TYPE in any kernel codes
//  you are using.
//
//#define TYPE    double
#define TYPE    float

// 
// we use the elapsed time timer from OpenMP
//
#include <omp.h>

//
//  Apple doesn't always put things in the standard places
//
#include <stdio.h>
#ifdef APPLE
#include <stdlib.h>
#endif

//
// Constants used to set elements of constant matrices
// used in the matrix multiply tests, initialize values,
// and to set tolerance in equality with floats
//
#define AVAL    3.0
#define BVAL    5.0
#define BIG     10000000.0
#define SMALL   0.00000001
#define TOL     0.001

//
// remove the backslashes if you want the code to produce huge 
// amounts of intermediate results to support debugging.
//
//#define DEBUG 1

//
// Function prototypes used in our software that tests matrix
// operations (matrix multiple test bed and our jacobi solvers)
//

double errsqr (int Ndim, int Mdim, TYPE* C, TYPE* Cref);

void mm_clear (int Ndim, int Mdim, TYPE* C); 

void mm_print (int Ndim, int Mdim, TYPE* C); 

void init_const_matrix (int Ndim,  int Mdim,  int Pdim, 
                  TYPE *A, TYPE* B, TYPE* C);

void init_progression_matrix (int Ndim,  int Mdim,  int Pdim, 
                  TYPE *A, TYPE* B, TYPE* C);
 
void output_results(int Ndim, int Mdim, int Pdim, 
                  int nerr, double ave_t, double min_t, double max_t);

void mm_tst_cases(int NTRIALS, int Ndim, int Mdim, int Pdim, TYPE* A, TYPE* B, 
        TYPE* C, void (*mm_func)(int, int, int, TYPE *, TYPE *, TYPE *));

void init_diag_dom_matrix(int Ndim,  TYPE *A);

void init_diag_dom_near_identity_matrix(int Ndim,  TYPE *A);

void init_colmaj_diag_dom_near_identity_matrix(int Ndim,  TYPE *A);
