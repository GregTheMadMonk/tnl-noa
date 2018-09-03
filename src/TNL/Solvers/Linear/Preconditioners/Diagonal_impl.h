/***************************************************************************
                          Diagonal_impl.h  -  description
                             -------------------
    begin                : Dec 17, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include "Diagonal.h"

namespace TNL {
namespace Solvers {
namespace Linear {
namespace Preconditioners {

#ifdef HAVE_CUDA
template< typename Real, typename Index, typename Matrix >
__global__ void matrixDiagonalToVectorKernel( const Matrix* matrix,
                                              Real* diagonal,
                                              Index size ) {
   Index elementIdx = blockDim. x * blockIdx. x + threadIdx. x;
   const Index maxGridSize = blockDim. x * gridDim. x;
   while( elementIdx < size )
   {
      diagonal[ elementIdx ] = matrix->getElementFast( elementIdx, elementIdx );
      elementIdx += maxGridSize;
   }
}

template< typename Real, typename Index >
__global__ void elementwiseVectorDivisionKernel( const Real* left,
                                                 const Real* right,
                                                 Real* result,
                                                 Index size )
{
   Index elementIdx = blockDim. x * blockIdx. x + threadIdx. x;
   const Index maxGridSize = blockDim. x * gridDim. x;
   while( elementIdx < size )
   {
      result[ elementIdx ] = left[ elementIdx ] / right[ elementIdx ];
      elementIdx += maxGridSize;
   }
}
#endif

template< typename Matrix >
void
Diagonal< Matrix >::
update( const Matrix& matrix )
{
//  std::cout << getType() << "->setMatrix()" << std::endl;

   TNL_ASSERT_GT( matrix.getRows(), 0, "empty matrix" );
   TNL_ASSERT_EQ( matrix.getRows(), matrix.getColumns(), "matrix must be square" );

   if( diagonal.getSize() != matrix.getRows() )
      diagonal.setSize( matrix.getRows() );

   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      for( int i = 0; i < diagonal.getSize(); i++ ) {
         diagonal[ i ] = matrix.getElement( i, i );
      }
   }
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      const IndexType& size = diagonal.getSize();
      dim3 cudaBlockSize( 256 );
      dim3 cudaBlocks;
      cudaBlocks.x = min( Devices::Cuda::getMaxGridSize(), Devices::Cuda::getNumberOfBlocks( size, cudaBlockSize.x ) );      

      Devices::Cuda::synchronizeDevice();
      matrixDiagonalToVectorKernel<<< cudaBlocks, cudaBlockSize >>>(
            &matrix.template getData< Devices::Cuda >(),
            diagonal.getData(),
            size );
      TNL_CHECK_CUDA_DEVICE;
#endif
   }
}

template< typename Matrix >
   template< typename Vector1, typename Vector2 >
bool
Diagonal< Matrix >::
solve( const Vector1& b, Vector2& x ) const
{
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      for( int i = 0; i < diagonal.getSize(); i++ ) {
         x[ i ] = b[ i ] / diagonal[ i ];
      }
   }
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      const IndexType& size = diagonal.getSize();
      dim3 cudaBlockSize( 256 );
      dim3 cudaBlocks;
      cudaBlocks.x = min( Devices::Cuda::getMaxGridSize(), Devices::Cuda::getNumberOfBlocks( size, cudaBlockSize.x ) );      

      elementwiseVectorDivisionKernel<<< cudaBlocks, cudaBlockSize >>>(
            b.getData(),
            diagonal.getData(),
            x.getData(),
            size );

      TNL_CHECK_CUDA_DEVICE;
#endif
   }
   return true;
}

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace TNL
