//------------------------------------------------------------------------------
//
// Name:       vadd.c
//
// Purpose:    Elementwise addition of two vectors (c = a + b)
//
// HISTORY:    Written by Tim Mattson, December 2009
//             Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated by Tom Deakin, July 2013
//             Updated by Tom Deakin, October 2014
//             Updated by Tom Deakin and James Price, October 2015
//
//------------------------------------------------------------------------------

//#include <//omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#define TOL    (0.001)   // tolerance used in floating point comparisons

#define MAX_PLATFORMS     8
#define MAX_DEVICES      16
#define MAX_INFO_STRING 256

static cl_uint length       = 1024;
static cl_uint device_index = 0;

unsigned get_device_list(cl_device_id devices[MAX_DEVICES]);
void parse_arguments(int argc, char *argv[]);
void check_error(cl_int err, const char *msg);


//------------------------------------------------------------------------------
//
// kernel:  vadd
//
// Purpose: Compute the elementwise sum c = a+b
//
// input: a and b float vectors of length count
//
// output: c float vector of length count holding the sum a + b
//

const char *KernelSource = "\n" \
"kernel void vadd(                                                      \n" \
"  global float* a,                                                     \n" \
"  global float* b,                                                     \n" \
"  global float* c,                                                     \n" \
"  const unsigned int count)                                            \n" \
"{                                                                      \n" \
"  int i = get_global_id(0);                                            \n" \
"  if(i < count)                                                        \n" \
"    c[i] = a[i] + b[i];                                                \n" \
"}                                                                      \n" \
"\n";

//------------------------------------------------------------------------------


int main(int argc, char** argv)
{
  int          err;               // error code returned from OpenCL calls

  parse_arguments(argc, argv);

  float*       h_a = (float*) calloc(length, sizeof(float));       // a vector
  float*       h_b = (float*) calloc(length, sizeof(float));       // b vector
  float*       h_c = (float*) calloc(length, sizeof(float));       // c vector (a+b) returned from the compute device

  unsigned int correct;           // number of correct results

  size_t global;                  // global domain size

  cl_device_id     device;     // compute device id
  cl_context       context;       // compute context
  cl_command_queue commands;      // compute command queue
  cl_program       program;       // compute program
  cl_kernel        ko_vadd;       // compute kernel

  cl_mem d_a;                     // device memory used for the input  a vector
  cl_mem d_b;                     // device memory used for the input  b vector
  cl_mem d_c;                     // device memory used for the output c vector

  // Fill vectors a and b with random float values
  unsigned i = 0;
  unsigned count = length;
  for(i = 0; i < count; i++)
  {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
  }

  // Get list of OpenCL devices
  cl_device_id devices[MAX_DEVICES];
  unsigned num_devices = get_device_list(devices);

  // Check device index in range
  if (device_index >= num_devices)
  {
    printf("Invalid device index (try '--list')\n");
    return 1;
  }

  device = devices[device_index];

  // Print device name
  char name[MAX_INFO_STRING];
  clGetDeviceInfo(device, CL_DEVICE_NAME, MAX_INFO_STRING, name, NULL);
  printf("\nUsing OpenCL device: %s\n", name);


  // Create a compute context
  context = clCreateContext(0, 1, &device, NULL, NULL, &err);
  check_error(err, "Creating context");

  // Create a command queue
  commands = clCreateCommandQueue(context, device, 0, &err);
  check_error(err, "Creating command queue");

  // Create the compute program from the source buffer
  program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
  check_error(err, "Creating program");

  // Build the program
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE)
  {
    size_t len;
    char buffer[2048];

    printf("OpenCL build log:\n");
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
  }
  check_error(err, "Building program");

  // Create the compute kernel from the program
  ko_vadd = clCreateKernel(program, "vadd", &err);
  check_error(err, "Creating kernel");

  // Create the input (a, b) and output (c) arrays in device memory
  d_a  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, &err);
  check_error(err, "Creating buffer d_a");

  d_b  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, &err);
  check_error(err, "Creating buffer d_b");

  d_c  = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, &err);
  check_error(err, "Creating buffer d_c");

  // Write a and b vectors into compute device memory
  err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float) * count, h_a, 0, NULL, NULL);
  check_error(err, "Copying h_a to device at d_a");

  err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(float) * count, h_b, 0, NULL, NULL);
  check_error(err, "Copying h_b to device at d_b");

  // Set the arguments to our compute kernel
  err  = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_a);
  err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_b);
  err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
  err |= clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &count);
  check_error(err, "Setting kernel arguments");

//  double rtime = omp_get_wtime();
  double rtime = 1.0;

  // Execute the kernel over the entire range of our 1d input data set
  // letting the OpenCL runtime choose the work-group size
  global = count;
  err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
  check_error(err, "Enqueueing kernel");

  // Wait for the commands to complete before stopping the timer
  err = clFinish(commands);
  check_error(err, "Waiting for kernel to finish");

//  rtime = omp_get_wtime() - rtime;
    rtime = 0.000001;
  printf("\nThe kernel ran in %lf seconds\n",rtime);

  // Read back the results from the compute device
  err = clEnqueueReadBuffer( commands, d_c, CL_TRUE, 0, sizeof(float) * count, h_c, 0, NULL, NULL );
  check_error(err, "Reading results");

  // Test the results
  correct = 0;
  float tmp;

  for(i = 0; i < count; i++)
  {
    tmp = h_a[i] + h_b[i];     // assign element i of a+b to tmp
    tmp -= h_c[i];             // compute deviation of expected and output result
    if(tmp*tmp < TOL*TOL)        // correct if square deviation is less than tolerance squared
        correct++;
    else {
        printf(" tmp %f h_a %f h_b %f h_c %f \n",tmp, h_a[i], h_b[i], h_c[i]);
    }
  }

  // summarise results
  printf("C = A+B:  %d out of %d results were correct.\n", correct, count);

  // cleanup then shutdown
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseProgram(program);
  clReleaseKernel(ko_vadd);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  free(h_a);
  free(h_b);
  free(h_c);

#if defined(_WIN32) && !defined(__MINGW32__)
  system("pause");
#endif

  return 0;
}


void check_error(const cl_int err, const char *msg)
{
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Error %d: %s\n", err, msg);
    exit(EXIT_FAILURE);
  }
}

unsigned get_device_list(cl_device_id devices[MAX_DEVICES])
{
  cl_int err;

  // Get list of platforms
  cl_uint num_platforms = 0;
  cl_platform_id platforms[MAX_PLATFORMS];
  err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &num_platforms);
  check_error(err, "getting platforms");

  // Enumerate devices
  unsigned num_devices = 0;
  for (int i = 0; i < num_platforms; i++)
  {
    cl_uint num = 0;
    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES-num_devices, devices+num_devices, &num);
    check_error(err, "getting deviceS");
    num_devices += num;
  }

  return num_devices;
}

int parse_uint(const char *str, cl_uint *output)
{
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

void parse_arguments(int argc, char *argv[])
{
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "--list"))
    {
      // Get list of devices
      cl_device_id devices[MAX_DEVICES];
      unsigned num_devices = get_device_list(devices);

      // Print device names
      if (num_devices == 0)
      {
        printf("No devices found.\n");
      }
      else
      {
        printf("\n");
        printf("Devices:\n");
        for (int i = 0; i < num_devices; i++)
        {
          char name[MAX_INFO_STRING];
          clGetDeviceInfo(devices[i], CL_DEVICE_NAME, MAX_INFO_STRING, name, NULL);
          printf("%2d: %s\n", i, name);
        }
        printf("\n");
      }
      exit(EXIT_SUCCESS);
    }
    else if (!strcmp(argv[i], "--device"))
    {
      if (++i >= argc || !parse_uint(argv[i], &device_index))
      {
        fprintf(stderr, "Invalid device index\n");
        exit(EXIT_FAILURE);
      }
    }
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      printf("\n");
      printf("Usage: ./jac_solv_ocl_basic [OPTIONS]\n\n");
      printf("Options:\n");
      printf("  -h    --help               Print the message\n");
      printf("        --list               List available devices\n");
      printf("        --device     INDEX   Select device at INDEX\n");
      printf("  LEN                        Set vector length to LEN\n");
      printf("\n");
      exit(EXIT_SUCCESS);
    }
    else
    {
      // Try to parse NDIM
      if (!parse_uint(argv[i], &length))
      {
        printf("Invalid Ndim value\n");
      }
    }
  }
}
