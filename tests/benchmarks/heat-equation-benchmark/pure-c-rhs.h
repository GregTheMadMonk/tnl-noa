/***************************************************************************
                          pure-c-rhs.h  -  description
                             -------------------
    begin                : Apr 2, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
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

#ifndef PURE_C_RHS_H
#define	PURE_C_RHS_H

#include<cuda.h>

/****
 * Just testing data for measuring performance
 * with different ways of passing data to kernels.
 */
struct Data
{
   double time, tau;
   tnlStaticVector< 2, double > c1, c2, c3, c4;
   tnlGrid< 2, double > grid;
};

#ifdef HAVE_CUDA

template< typename Real, typename Index >
__global__ void boundaryConditionsKernel( const Real* u, Real* aux,
                                          const Index gridXSize, const Index gridYSize )
{
   const Index i = ( blockIdx.x ) * blockDim.x + threadIdx.x;
   const Index j = ( blockIdx.y ) * blockDim.y + threadIdx.y;
   if( i == 0 && j < gridYSize )
      aux[ j * gridXSize ] = 0.0; //u[ j * gridXSize + 1 ];
   if( i == gridXSize - 1 && j < gridYSize )
      aux[ j * gridXSize + gridXSize - 2 ] = 0.0; //u[ j * gridXSize + gridXSize - 1 ];      
   if( j == 0 && i < gridXSize )
      aux[ j * gridXSize ] = 0.0; //u[ j * gridXSize + 1 ];
   if( j == gridYSize -1  && i < gridXSize )
      aux[ j * gridXSize + gridXSize - 2 ] = 0.0; //u[ j * gridXSize + gridXSize - 1 ];      
    
}


template< typename Real, typename Index >
__global__ void heatEquationKernel( const Real* u, 
                                    Real* aux,
                                    const Real tau,
                                    const Real hx_inv,
                                    const Real hy_inv,
                                    const Index gridXSize,
                                    const Index gridYSize,
                                    Data d1,
                                    Data d2 )
{
   const Index i = blockIdx.x * blockDim.x + threadIdx.x;
   const Index j = blockIdx.y * blockDim.y + threadIdx.y;
   if( i > 0 && i < gridXSize - 1 &&
       j > 0 && j < gridYSize - 1 )
   {
      const Index c = j * gridXSize + i;
      aux[ c ] = tau * ( ( u[ c - 1 ] - 2.0 * u[ c ] + u[ c + 1 ] ) * hx_inv +
                       ( u[ c - gridXSize ] - 2.0 * u[ c ] + u[ c + gridXSize ] ) * hy_inv );
   }
}

template< typename RealType >
bool pureCRhsCuda( dim3 cudaGridSize,
                   dim3 cudaBlockSize,
                   RealType* cuda_u,
                   RealType* cuda_aux,
                   const RealType& tau,
                   const RealType& hx_inv,
                   const RealType& hy_inv,
                   int gridXSize,
                   int gridYSize )
{
   /*Real* kernelTime = tnlCuda::passToDevice( time );
   Real* kernelTau = tnlCuda::passToDevice( tau );
   typedef tnlStaticVector< 2, Real > Coordinates;
   Coordinates c;
   Coordinates* kernelC1 = tnlCuda::passToDevice( c );
   Coordinates* kernelC2 = tnlCuda::passToDevice( c );
   Coordinates* kernelC3 = tnlCuda::passToDevice( c );
   Coordinates* kernelC4 = tnlCuda::passToDevice( c );
   typedef tnlGrid< 2, Real, tnlCuda, int > Grid;
   Grid g;
   Grid* kernelGrid = tnlCuda::passToDevice( g );*/
   Data d, d2;
   //Data* kernelD = tnlCuda::passToDevice( d );

   int cudaErr;
   /****
    * Neumann boundary conditions
    */
   //cout << "Setting boundary conditions ... " << endl;
   boundaryConditionsKernel<<< cudaGridSize, cudaBlockSize >>>( cuda_u, cuda_aux, gridXSize, gridYSize );
   if( ( cudaErr = cudaGetLastError() ) != cudaSuccess )
   {
      cerr << "Setting of boundary conditions failed. " << cudaErr << endl;
      return false;
   }

   /****
    * Laplace operator
    */
   //cout << "Laplace operator ... " << endl;
   heatEquationKernel<<< cudaGridSize, cudaBlockSize >>>
      ( cuda_u, cuda_aux, tau, hx_inv, hy_inv, gridXSize, gridYSize, d, d2 );
   if( cudaGetLastError() != cudaSuccess )
   {
      cerr << "Laplace operator failed." << endl;
      return false;
   }

   //tnlCuda::freeFromDevice( kernelD );
   /*tnlCuda::freeFromDevice( kernelTau );
   tnlCuda::freeFromDevice( kernelC1 );
   tnlCuda::freeFromDevice( kernelC2 );
   tnlCuda::freeFromDevice( kernelC3 );
   tnlCuda::freeFromDevice( kernelC4 );
   tnlCuda::freeFromDevice( kernelGrid );*/

   return true;
}

#endif

#endif	/* PURE_C_RHS_H */

