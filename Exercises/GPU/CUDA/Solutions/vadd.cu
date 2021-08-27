//------------------------------------------------------------------------------
//
// Name:            vadd_cpp.cpp
// 
// Purpose:     Elementwise addition of two vectors (c = a + b)
//
//                          c = a + b
//
// HISTORY:     Written by Tim Mattson, June 2011
//                  Ported to C++ Wrapper API by Benedict Gaster, September 2011
//                  Updated to C++ Wrapper API v1.2 by Tom Deakin and Simon McIntosh-Smith, October 2012
//                  Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//                  Ported to CUDA by Francesco Rossi, Oct 2013
//------------------------------------------------------------------------------

#define __CL_ENABLE_EXCEPTIONS

#include "util.hpp" // utility library

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <iostream>
#include <fstream>

#include "cuda_util.hpp" //for CudaSafeCall

//------------------------------------------------------------------------------

#define TOL     (0.001)  // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c

//CUDA kernel
__global__ void vadd_kernel(const float* a, const float* b, float* c, const int count)
{
    const int i = blockDim.x*blockIdx.x + threadIdx.x; //equiv to "int i = get_global_id(0);" for 1D launches                 
    if(i < count)	//Out of bounds guard
    {
        c[i] = a[i] + b[i];                     
    }
}

int main(void)
{
    std::vector<float> h_a(LENGTH);                  // a vector 
    std::vector<float> h_b(LENGTH);                  // b vector    
    std::vector<float> h_c (LENGTH, 0xdeadbeef); // c = a + b, from compute device
    
    float* d_a;     // device memory used for the input  a vector
    float* d_b;     // device memory used for the input  b vector
    float* d_c;     // device memory used for the output c vector
    
     // Fill vectors a and b with random float values
    int count = LENGTH;
    for(int i = 0; i < count; i++)
    {
        h_a[i]   = rand() / (float)RAND_MAX;
        h_b[i]   = rand() / (float)RAND_MAX;
    }

    //CUDA part begin
    CudaSafeCall( cudaSetDevice(0) );
    
    CudaSafeCall( cudaMalloc(&d_a, sizeof(float)*LENGTH) ); // allocates device memory
    CudaSafeCall( cudaMalloc(&d_b, sizeof(float)*LENGTH) );
    CudaSafeCall( cudaMalloc(&d_c, sizeof(float)*LENGTH) );
    
    CudaSafeCall( cudaMemcpy(d_a, &h_a[0], sizeof(float)*LENGTH, cudaMemcpyHostToDevice) ); //copies vectors initialized on host to device
    CudaSafeCall( cudaMemcpy(d_b, &h_b[0], sizeof(float)*LENGTH, cudaMemcpyHostToDevice) );
    
    cudaEvent_t start, stop;    //just for timing purpose
    float elapsedTime;          //just for timing purpose
    cudaEventCreate(&start);    //just for timing purpose
    cudaEventCreate(&stop);     //just for timing purpose
    cudaEventRecord(start,0);       //just for timing purpose

    vadd_kernel<<<(LENGTH-1)/512+1,512>>>(d_a,d_b,d_c,LENGTH); //vadd_kernel kernel launch, (LENGTH-1)/512+1 is a division by 512 rounding up
    
    cudaEventRecord(stop,0);                        //just for timing purpose
    cudaEventSynchronize(stop);                 //just for timing purpose
    cudaEventElapsedTime(&elapsedTime, start, stop); //just for timing purpose
    printf("\nThe kernels ran in %lf seconds\n", elapsedTime/1000.f);

    CudaSafeCall( cudaMemcpy(&h_c[0], d_c, sizeof(float)*LENGTH, cudaMemcpyDeviceToHost) ); //copies vector computed on device to host
    
    CudaSafeCall( cudaFree(d_a) ); //frees up memory
    CudaSafeCall( cudaFree(d_b) );
    CudaSafeCall( cudaFree(d_c) );
    
    CudaSafeCall( cudaEventDestroy(start) );
	CudaSafeCall( cudaEventDestroy(stop) );
	
    //CUDA part end
    
    // Test the results
    int correct = 0;
    float tmp;
    for(int i = 0; i < count; i++) {
        tmp = h_a[i] + h_b[i]; // expected value for d_c[i]
        tmp -= h_c[i];                              // compute errors
        if(tmp*tmp < TOL*TOL) {       // correct if square deviation is less 
            correct++;                                //    than tolerance squared
        }
        else {

            printf(
                " tmp %f h_a %f h_b %f  h_c %f \n",
                tmp, 
                h_a[i], 
                h_b[i], 
                h_c[i]);
        }
    }
    // summarize results
	printf(
                "vector add to find C = A+B:    %d out of %d results were correct.\n", 
                correct, 
                count);
    
    return 0;
}
