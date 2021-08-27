/*

This program will numerically compute the integral of

                  4/(1+x*x) 
				  
from 0 to 1.  The value of this integral is pi -- which 
is great since it gives us an easy way to check the answer.

This is the sequenctial version of the program.  It uses
the OpenMP timer.

History: Written by Tim Mattson, 11/99.

*/
#include <stdio.h>
#include <omp.h>
static long num_steps = 100000000;
double step;
int main (int argc, char **argv)
{
   int i;
   double x, pi, sum = 0.0;
   double start_time, run_time;

   step = 1.0/(double) num_steps;

   start_time = omp_get_wtime();

   for (i=0;i<= num_steps; i++){
      x = (i+0.5)*step;
      sum = sum + 4.0/(1.0+x*x);
   }

   pi = step * sum;
   run_time = omp_get_wtime() - start_time;
   printf("\n pi with %ld steps is %lf in %lf seconds\n ",
                          num_steps,pi,run_time);
}	  
