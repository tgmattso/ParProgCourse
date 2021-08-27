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
//           Modified by Simon McIntosh-Smith, September 2011
//           Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//           Ported to CUDA by F.Rossi, Oct 2013
//------------------------------------------------------------------------------

#include "matmul.hpp"
#include "matrix_lib.hpp"

#include "cuda_util.hpp" //for CudaSafeCall

//Naive dense matrix multiplication kernel.
//
//An optimized and documented version of the parallel algorithm
//is available in section 3.2 of http://docs.nvidia.com/cuda/cuda-c-programming-guide/
//
__global__ void kernel_mmul(
    const int Mdim,
    const int Ndim,
    const int Pdim,
    const float* A, 
    const float* B,
    float* C) 
{ 
    int k;
    int i = blockDim.x*blockIdx.x+threadIdx.x;
    int j = blockDim.y*blockIdx.y+threadIdx.y;
    float tmp;
    if ( (i < Ndim) && (j <Mdim)) 
    {
        tmp = 0.0;
        for(k=0;k<Pdim;k++)
            tmp += A[i*Ndim+k] * B[k*Pdim+j]; 
        C[i*Ndim+j] = tmp;
    }
}

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

    std::vector<float> h_A(szA); // Host memory for Matrix A
    std::vector<float> h_B(szB); // Host memory for Matrix B
    std::vector<float> h_C(szC); // Host memory for Matrix C

    float* d_a;     // device memory used for the input  a vector
    float* d_b;     // device memory used for the input  b vector
    float* d_c;     // device memory used for the output c vector
   
    initmat(Mdim, Ndim, Pdim, h_A, h_B, h_C);

    printf("\n===== Sequential, matrix mult (dot prod), order %d on host CPU ======\n",ORDER);
    for(int i = 0; i < COUNT; i++)
    {
        zero_mat(Ndim, Mdim, h_C);
        start_time = wtime();

        seq_mat_mul_sdot(Mdim, Ndim, Pdim, h_A, h_B, h_C);

        run_time  = wtime() - start_time;
        results(Mdim, Ndim, Pdim, h_C, run_time);
    }
    
    //CUDA part begin
    CudaSafeCall( cudaSetDevice(0) );
    
    CudaSafeCall( cudaMalloc(&d_a, sizeof(float)*szA) ); // allocates device memory
    CudaSafeCall( cudaMalloc(&d_b, sizeof(float)*szB) );
    CudaSafeCall( cudaMalloc(&d_c, sizeof(float)*szC) );
    
    CudaSafeCall( cudaMemcpy(d_a, &h_A[0], sizeof(float)*szA, cudaMemcpyHostToDevice) ); //copies vectors initialized on host to device
    CudaSafeCall( cudaMemcpy(d_b, &h_B[0], sizeof(float)*szB, cudaMemcpyHostToDevice) );
    
    cudaEvent_t start, stop;    //just for timing purpose
    float elapsed_time = 0;          //just for timing purpose
    cudaEventCreate(&start);    //just for timing purpose
    cudaEventCreate(&stop);     //just for timing purpose
    printf("\n===== CUDA, matrix mult, C(i,j) per work item, order %d ======\n",Ndim);

    // Do the multiplication COUNT times
    for (int i = 0; i < COUNT; i++)
    {
        zero_mat(Ndim, Mdim, h_C);

        dim3 grid,block;
        
        block.x = 32;
        block.y = 16;
        block.z = 1;
        
        grid.x  = (Ndim-1)/block.x+1; //round up division by block.x
        grid.y  = (Mdim-1)/block.y+1;
        grid.z  = 1;
        
        cudaEventRecord(start,0);       //just for timing purpose

        kernel_mmul<<<grid,block>>>(Mdim,Ndim,Pdim,d_a,d_b,d_c);
        
        cudaEventRecord(stop,0);                        //just for timing purpose
        cudaEventSynchronize(stop);                     //just for timing purpose
        cudaEventElapsedTime(&elapsed_time, start, stop);   //just for timing purpose
        
        CudaSafeCall( cudaMemcpy(&h_C[0], d_c, sizeof(float)*szC, cudaMemcpyDeviceToHost) ); //copies vector computed on device to host
        
        results(Mdim, Ndim, Pdim, h_C, elapsed_time/1000.f);
    }
    
    CudaSafeCall( cudaFree(d_a) ); //frees up memory
    CudaSafeCall( cudaFree(d_b) );
    CudaSafeCall( cudaFree(d_c) );
    
    CudaSafeCall( cudaEventDestroy(start) );
    CudaSafeCall( cudaEventDestroy(stop) );
    
    //cuda part end
    
    return EXIT_SUCCESS;
}
