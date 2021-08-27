//------------------------------------------------------------------------------
//
// Name:       vadd_cpp.cpp
// 
// Purpose:    Elementwise addition of two vectors (c = a + b)
//
//                   c = a + b
//
// HISTORY:    Written by Tim Mattson, Oct 2013
//             
//------------------------------------------------------------------------------

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <iostream>
#include <fstream>

//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c
extern  double wtime();   // timer routine in CPP common

int main(void)
{
    std::vector<float> a(LENGTH);              // a vector 
    std::vector<float> b(LENGTH);              // b vector 	
    std::vector<float> c(LENGTH, 0xdeadbeef);  // c vector ... c will = a + b

    // Fill vectors a and b with random float values
    int count = LENGTH;
    for(int i = 0; i < count; i++)
    {
       a[i]  = rand() / (float)RAND_MAX;
       b[i]  = rand() / (float)RAND_MAX;
    }

    double t0 = wtime();

    // Compute the vector sum c = a + b
    for(int i = 0; i < count; i++)
    {
       c[i]  = a[i] + b[i];
    }

    double rtime = wtime() - t0;
    printf("\nThe kernels ran in %lf seconds\n", rtime);

    // Test the results
    int correct = 0;
    float tmp;
    for(int i = 0; i < count; i++) {
       tmp = a[i] + b[i];        // Expected value for c[i]
       tmp -= c[i];              // Compute errors
       if(tmp*tmp < TOL*TOL) {   // Correct if square deviation is less 
          correct++;             //        than tolerance squared
       }
       else {
          printf( " tmp %d %f a %f b %f  c %f \n",
                    i,tmp, a[i], b[i], c[i]);
       }
    }

    // summarize results
    printf( "vector add to find C = A+B:  %d out of %d results were correct.\n", 
            correct, count);
}
