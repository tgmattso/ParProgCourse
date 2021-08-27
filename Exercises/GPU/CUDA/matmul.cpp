//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix Multipliplication driver
//
//  PURPOSE: This is a driver program to test various ways of computing
//           the product:
//
//                C  = A * B
//
//           A and B are set to constant matrices so we
//           can make a quick test of the multiplication.
//
//  USAGE:   The matrices are constant matrices, square and the order is
//           set as a constant, ORDER (see mult.h).
//
//  HISTORY: Written by Tim Mattson, August 2010 
//
//------------------------------------------------------------------------------

#include "matmul.hpp"
#include "matrix_lib.hpp"

int main(void)
{

    int Mdim, Ndim, Pdim;   // A[N][P], B[P][M], C[N][M]
    int szA, szB, szC;      // number of elements in each matrix

    double start_time;      // Starting time
    double run_time;        // timing data

    Ndim = ORDER;
    Pdim = ORDER;
    Mdim = ORDER;

    szA = Ndim * Pdim;
    szB = Pdim * Mdim;
    szC = Ndim * Mdim;

    std::vector<float> A(szA); // Host memory for Matrix A
    std::vector<float> B(szB); // Host memory for Matrix B
    std::vector<float> C(szC); // Host memory for Matrix C

    initmat(Mdim, Ndim, Pdim, A, B, C);

    printf("\n===== Sequential, matrix mult (dot prod), order %d on host CPU ======\n",ORDER);
    float tmp;
    zero_mat(Ndim, Mdim, C);
    start_time = wtime();

    for (int ii = 0; ii < Ndim; ii++) {
      for (int jj = 0; jj < Mdim; jj++) {
         tmp = 0.0f;
         for (int kk = 0; kk < Pdim; kk++) {
             /* C(ii,jj) = sum(over kk) A(ii,kk) * B(kk,jj) */
             tmp += A[ii*Ndim+kk] * B[kk*Pdim+jj];
         }
         C[ii*Ndim+jj] = tmp;
      }
    }

    run_time  = wtime() - start_time;
    results(Mdim, Ndim, Pdim, C, run_time);

    return EXIT_SUCCESS;
}

