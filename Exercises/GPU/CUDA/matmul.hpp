//------------------------------------------------------------------------------
//
//  Include fle for the Matrix Multiply test harness
//
//  HISTORY: Written by Tim Mattson, August 2010
//
//------------------------------------------------------------------------------

#ifndef __MULT_HDR
#define __MULT_HDR

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include <vector>

#include "matrix_lib.hpp"

//------------------------------------------------------------------------------
//  functions from ../Common
//------------------------------------------------------------------------------
extern double wtime();   // returns time since some fixed past point (wtime.c)

//------------------------------------------------------------------------------
//  Constants
//------------------------------------------------------------------------------
#define ORDER    1024    // Order of the square matrices A, B, and C
#define AVAL     3.0     // A elements are constant and equal to AVAL
#define BVAL     5.0     // B elements are constant and equal to BVAL
#define TOL      (0.001) // tolerance used in floating point comparisons
#define DIM      2       // Max dim for NDRange
#define COUNT    1       // number of times to do each multiplication
#define SUCCESS  1
#define FAILURE  0

#endif
