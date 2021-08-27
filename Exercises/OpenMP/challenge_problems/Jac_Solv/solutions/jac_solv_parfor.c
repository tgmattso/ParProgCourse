/*
**  PROGRAM: jacobi Solver .. parallel For version
**
**  PURPOSE: This program will explore use of a jacobi iterative
**           method to solve a system of linear equations (Ax= b).
** 
**           Here is the basic idea behind the method.   Rewrite 
**           the matrix A as a Lower Triangular (L), upper triangular
**           (U) and diagonal matrix (D)
**
**                Ax = (L + D + U)x = b
**
**            Carry out the multiplication and rearrange:
**
**                Dx = b - (L+U)x  -->   x = (b-(L+U)x)/D
**
**           We can do this iteratively
**
**                x_new = (b-(L+U)x_old)/D
**
**  USAGE:   Run wtihout arguments to use default SIZE.
**
**              ./jac_solv
**
**           Run with a single argument for the order of the A
**           matrix ... for example
**
**              ./jac_solv 2500
**
**  HISTORY: Written by Tim Mattson, Oct 2015
*/

#include<omp.h>
#include<math.h>
#include "mm_utils.h"   //a library of basic matrix utilities functions
                        //and some key constants used in this program 
                        //(such as TYPE)

#define TOLERANCE 0.001
#define DEF_SIZE  1000
#define MAX_ITERS 5000
#define LARGE     1000000.0

//#define DEBUG    1     // output a small subset of intermediate values
//#define VERBOSE  1     

int main(int argc, char **argv)
{
   int Ndim;           // A[Ndim][Ndim]
   int i,j, iters;
   double start_time, elapsed_time;
   TYPE conv, tmp, err, chksum;
   TYPE *A, *b, *x1, *x2, *xnew, *xold, *xtmp; 

// set matrix dimensions and allocate memory for matrices
   if(argc ==2){
      Ndim = atoi(argv[1]);
   }
   else{
      Ndim = DEF_SIZE;
   }

   printf(" jacobi solver parallel for version: ndim = %d\n",Ndim);

   A    = (TYPE *) malloc(Ndim*Ndim*sizeof(TYPE));
   b    = (TYPE *) malloc(Ndim*sizeof(TYPE));
   x1   = (TYPE *) malloc(Ndim*sizeof(TYPE));
   x2   = (TYPE *) malloc(Ndim*sizeof(TYPE));

   if (!A || !b || !x1 || !x2)
   {
        printf("\n memory allocation error\n");
        exit(-1);
   }

   // generate our diagonally dominant matrix, A
   init_diag_dom_near_identity_matrix(Ndim, A);

#ifdef VERBOSE
   mm_print(Ndim, Ndim, A);
#endif

//
// Initialize x and just give b some non-zero random values
//
   for(i=0; i<Ndim; i++){
     x1[i] = (TYPE)0.0;
     x2[i] = (TYPE)0.0;
     b[i]  = (TYPE)(rand()%51)/100.0;
   }

   start_time = omp_get_wtime();
// 
// jacobi iterative solver
//
   conv  = LARGE;
   iters = 0;
   xnew  = x1;
   xold  = x2;

   {
   // note: i am comparing against the convergence sqaured.  This saves a
   // sqrt and an extra barrier.
   while((conv > TOLERANCE*TOLERANCE) && (iters<MAX_ITERS))
   {
     {
        iters++;
        conv = 0.0;
        xtmp  = xnew;   // don't copy arrays.
        xnew  = xold;   // just swap pointers.
        xold  = xtmp;
     }

     #pragma omp parallel for private(i,j)
     for (i=0; i<Ndim; i++){
         xnew[i] = (TYPE) 0.0;
         for (j=0; j<Ndim;j++){
         //    if(i!=j)
         //      xnew[i]+= A[i*Ndim + j]*xold[j];
               xnew[i]+= A[i*Ndim + j]*xold[j] * (i != j);
         }
         xnew[i] = (b[i]-xnew[i])/A[i*Ndim+i];

     }
     //  
     // test convergence
     //
     #pragma omp parallel for private(tmp) reduction(+:conv)
     for (i=0; i<Ndim; i++){
         tmp  = xnew[i]-xold[i];
         conv += tmp*tmp;
     }
#ifdef DEBUG
     printf(" conv = %f \n",(float)conv);
#endif

   }
   }
   conv = sqrt((double)conv);
   elapsed_time = omp_get_wtime() - start_time;
   printf(" Convergence = %g with %d iterations and %f seconds\n",
         (float)conv, iters, (float)elapsed_time);
   
   //
   // test answer by multiplying my computed value of x by
   // the input A matrix and comparing the result with the 
   // input b vector.
   //
   err    = (TYPE) 0.0;
   chksum = (TYPE) 0.0;

   for(i=0;i<Ndim;i++){
      xold[i] = (TYPE) 0.0;
      for(j=0; j<Ndim; j++)
         xold[i] += A[i*Ndim+j]*xnew[j];
      tmp = xold[i] - b[i];
#ifdef DEBUG
      printf(" i=%d, diff = %f,  computed b = %f, input b= %f \n",
                    i, (float)tmp, (float)xold[i], (float)b[i]);
#endif
      chksum += xnew[i];
      err += tmp*tmp;
   }
   err = sqrt((double)err);
   printf("jacobi solver: err = %f, solution checksum = %f \n",
                               (float)sqrt(err), (float)chksum);

  free(A);
  free(b);
  free(x1);
  free(x2);
}
