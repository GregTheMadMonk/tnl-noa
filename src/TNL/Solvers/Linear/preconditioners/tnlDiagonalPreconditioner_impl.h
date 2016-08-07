
#pragma once

#include "tnlDiagonalPreconditioner.h"

namespace TNL {
namespace Solvers {
namespace Linear {   

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

template< typename Real, typename Device, typename Index >
   template< typename MatrixPointer >
void
tnlDiagonalPreconditioner< Real, Device, Index >::
update( const MatrixPointer& matrix )
{
//  std::cout << getType() << "->setMatrix()" << std::endl;

   Assert( matrix->getRows() > 0 && matrix->getRows() == matrix->getColumns(), );

   if( diagonal.getSize() != matrix->getRows() )
      diagonal.setSize( matrix->getRows() );

   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      for( int i = 0; i < diagonal.getSize(); i++ ) {
         diagonal[ i ] = matrix->getElement( i, i );
      }
   }
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      //Matrix* kernelMatrix = tnlCuda::passToDevice( matrix );

      const Index& size = diagonal.getSize();
      dim3 cudaBlockSize( 256 );
      dim3 cudaBlocks;
      cudaBlocks.x = min( Devices::Cuda::getMaxGridSize(), Devices::Cuda::getNumberOfBlocks( size, cudaBlockSize.x ) );      

      Devices::Cuda::synchronizeDevice();
      matrixDiagonalToVectorKernel<<< cudaBlocks, cudaBlockSize >>>(
            &matrix.template getData< Devices::Cuda >(),
            diagonal.getData(),
            size );

      checkCudaDevice;
      //tnlCuda::freeFromDevice( kernelMatrix );
      //checkCudaDevice;
#endif
   }
}

template< typename Real, typename Device, typename Index >
   template< typename Vector1, typename Vector2 >
bool
tnlDiagonalPreconditioner< Real, Device, Index >::
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
      const Index& size = diagonal.getSize();
      dim3 cudaBlockSize( 256 );
      dim3 cudaBlocks;
      cudaBlocks.x = min( Devices::Cuda::getMaxGridSize(), Devices::Cuda::getNumberOfBlocks( size, cudaBlockSize.x ) );      

      elementwiseVectorDivisionKernel<<< cudaBlocks, cudaBlockSize >>>(
            b.getData(),
            diagonal.getData(),
            x.getData(),
            size );

      checkCudaDevice;
#endif
   }
   return true;
}

} // namespace Linear
} // namespace Solvers
} // namespace TNL
