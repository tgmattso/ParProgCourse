
//#define TYPE double
#define TYPE float

#if (TYPE == double)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

kernel void jacobi(
  const unsigned Ndim,
  global const TYPE * restrict A,
  global const TYPE * restrict b,
  global const TYPE * restrict xold,
  global TYPE * restrict xnew)
{
  size_t i = get_global_id(0);

  xnew[i] = (TYPE) 0.0;
  for (int j = 0; j < Ndim;)
  {
    xnew[i] += A[j*Ndim + i] * xold[j] * (TYPE)(i != j); j++;
    xnew[i] += A[j*Ndim + i] * xold[j] * (TYPE)(i != j); j++;
    xnew[i] += A[j*Ndim + i] * xold[j] * (TYPE)(i != j); j++;
    xnew[i] += A[j*Ndim + i] * xold[j] * (TYPE)(i != j); j++;
  }
  xnew[i] = (b[i] - xnew[i]) / A[i*Ndim + i];
}


kernel void convergence(
  global const TYPE * restrict xold,
  global const TYPE * restrict xnew,
  local TYPE * restrict conv_loc,
  global TYPE * restrict conv
  )
{
  size_t i = get_global_id(0);
  TYPE tmp;

  tmp = xnew[i] - xold[i];
  conv_loc[get_local_id(0)] = tmp * tmp;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int offset = get_local_size(0) / 2; offset > 0; offset /= 2)
  {
    if (get_local_id(0) < offset)
    {
      conv_loc[get_local_id(0)] += conv_loc[get_local_id(0) + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (get_local_id(0) == 0)
  {
    conv[get_group_id(0)] = conv_loc[0];
  }
}
