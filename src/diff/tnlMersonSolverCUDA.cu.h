/***************************************************************************
                          tnlMersonSolverCUDATester.h
                             -------------------
    begin                : Jan 13, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#ifndef TNLMERSONSOLVERCUDA_CU_H_
#define TNLMERSONSOLVERCUDA_CU_H_

#ifdef HAVE_CUDA

template< class T > __global__ void computeK2ArgKernel( const int size, const T tau, const T* u, const T* k1, T* k2_arg )
{
   int i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k2_arg[ i ] = u[ i ] + tau * ( 1.0 / 3.0 * k1[ i ] );
}

template< class T > __global__ void computeK3ArgKernel( const int size, const T tau, const T* u, const T* k1, const T* k2, T* k3_arg )
{
   int i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k3_arg[ i ] = u[ i ] + tau * 1.0 / 6.0 * ( k1[ i ] + k2[ i ] );
}

template< class T > __global__ void computeK4ArgKernel( const int size, const T tau, const T* u, const T* k1, const T* k3, T* k4_arg )
{
   int i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k4_arg[ i ] = u[ i ] + tau * ( 0.125 * k1[ i ] + 0.375 * k3[ i ] );
}

template< class T > __global__ void computeK5ArgKernel( const int size, const T tau, const T* u, const T* k1, const T* k3, const T* k4, T* k5_arg )
{
   int i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k5_arg[ i ] = u[ i ] + tau * ( 0.5 * k1[ i ] - 1.5 * k3[ i ] + 2.0 * k4[ i ] );
}

template< class T > __global__ void computeErrKernel( const int size, const T tau, const T* k1, const T* k3, const T* k4, const T* k5, T* err )
{
   int i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      err[ i ] = 1.0 / 3.0 *  tau * fabs( 0.2 * k1[ i ] +
                                         -0.9 * k3[ i ] +
                                          0.8 * k4[ i ] +
                                         -0.1 * k5[ i ] );
}

template< class T > __global__ void updateUKernel( const int size, const T tau, const T* k1, const T* k4, const T* k5, T* u )
{
        int i = blockIdx. x * blockDim. x + threadIdx. x;
        if( i < size )
                u[ i ] += tau / 6.0 * ( k1[ i ] + 4.0 * k4[ i ] + k5[ i ] );
}

#endif

#endif /* TNLMERSONSOLVERCUDA_CU_H_ */
