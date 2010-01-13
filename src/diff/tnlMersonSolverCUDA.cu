/***************************************************************************
                          tnlMersonSolverCUDA.cu  -  description
                             -------------------
    begin                : Nov 21, 2009
    copyright            : (C) 2009 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 
#include <diff/tnlMersonSolverCUDA.cu.h>


void computeK2Arg( const int size, 
                   const int block_size,
                   const int grid_size,
                   const float tau,
                   const float* u,
                   const float* k1,
                   float* k2_arg )
{
   computeK2ArgKernel<<< grid_size, block_size >>>( size, tau, u, k1, k2_arg );
}

void computeK2Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const double tau,
                   const double* u,
                   const double* k1,
                   double* k2_arg )
{
   computeK2ArgKernel<<< grid_size, block_size >>>( size, tau, u, k1, k2_arg );
}

void computeK3Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const float tau,
                   const float* u,
                   const float* k1,
                   const float* k2,
                   float* k3_arg )
{
   computeK3ArgKernel<<< grid_size, block_size >>>( size, tau, u, k1, k2, k3_arg );
}

void computeK3Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const double tau,
                   const double* u,
                   const double* k1,
                   const double* k2,
                   double* k3_arg )
{
   computeK3ArgKernel<<< grid_size, block_size >>>( size, tau, u, k1, k2, k3_arg );
}

void computeK4Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const float tau,
                   const float* u,
                   const float* k1,
                   const float* k3,
                   float* k4_arg )
{
   computeK4ArgKernel<<< grid_size, block_size >>>( size, tau, u, k1, k3, k4_arg );
}

void computeK4Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const double tau,
                   const double* u,
                   const double* k1,
                   const double* k3,
                   double* k4_arg )
{
   computeK4ArgKernel<<< grid_size, block_size >>>( size, tau, u, k1, k3, k4_arg );
}

void computeK5Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const float tau,
                   const float* u,
                   const float* k1,
                   const float* k3,
                   const float* k4,
                   float* k5_arg )
{
   computeK5ArgKernel<<< grid_size, block_size >>>( size, tau, u, k1, k3, k4, k5_arg );
}

void computeK5Arg( const int size,
                   const int block_size,
                   const int grid_size,
                   const double tau,
                   const double* u,
                   const double* k1,
                   const double* k3,
                   const double* k4,
                   double* k5_arg )
{
   computeK5ArgKernel<<< grid_size, block_size >>>( size, tau, u, k1, k3, k4, k5_arg );
}

void updateU( const int size,
              const int block_size,
              const int grid_size,
              const float tau,
              const float* k1,
              const float* k4,
              const float* k5,
              float* u )
{
   updateUKernel<<< grid_size, block_size >>>( size, tau, k1, k4, k5, u );
}

void updateU( const int size,
              const int block_size,
              const int grid_size,
              const double tau,
              const double* k1,
              const double* k4,
              const double* k5,
              double* u )
{
   updateUKernel<<< grid_size, block_size >>>( size, tau, k1, k4, k5, u );
}