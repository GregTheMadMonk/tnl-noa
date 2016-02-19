#ifndef TNLDIAGONALPRECONDITIONER_IMPL_H_
#define TNLDIAGONALPRECONDITIONER_IMPL_H_

#include "tnlDiagonalPreconditioner.h"

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
   template< typename Matrix >
void
tnlDiagonalPreconditioner< Real, Device, Index >::
update( const Matrix& matrix )
{
//   cout << getType() << "->setMatrix()" << endl;

   tnlAssert( matrix.getRows() > 0 && matrix.getRows() == matrix.getColumns(), );

   if( diagonal.getSize() != matrix.getRows() )
      diagonal.setSize( matrix.getRows() );

   if( ( tnlDeviceEnum ) DeviceType::DeviceType == tnlHostDevice )
   {
      for( int i = 0; i < diagonal.getSize(); i++ ) {
         diagonal[ i ] = matrix.getElement( i, i );
      }
   }
   if( ( tnlDeviceEnum ) DeviceType::DeviceType == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      Matrix* kernelMatrix = tnlCuda::passToDevice( matrix );

      const Index& size = diagonal.getSize();
      dim3 cudaBlockSize( 256 );
      dim3 cudaBlocks;
      cudaBlocks.x = Min( tnlCuda::getMaxGridSize(), tnlCuda::getNumberOfBlocks( size, cudaBlockSize.x ) );      

      matrixDiagonalToVectorKernel<<< cudaBlocks, cudaBlockSize >>>(
            kernelMatrix,
            diagonal.getData(),
            size );

      checkCudaDevice;
      tnlCuda::freeFromDevice( kernelMatrix );
      checkCudaDevice;
#endif
   }
}

template< typename Real, typename Device, typename Index >
   template< typename Vector1, typename Vector2 >
bool
tnlDiagonalPreconditioner< Real, Device, Index >::
solve( const Vector1& b, Vector2& x ) const
{
//   cout << getType() << "->solve()" << endl;
   if( ( tnlDeviceEnum ) DeviceType::DeviceType == tnlHostDevice )
   {
      for( int i = 0; i < diagonal.getSize(); i++ ) {
         x[ i ] = b[ i ] / diagonal[ i ];
      }
   }
   if( ( tnlDeviceEnum ) DeviceType::DeviceType == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      const Index& size = diagonal.getSize();
      dim3 cudaBlockSize( 256 );
      dim3 cudaBlocks;
      cudaBlocks.x = Min( tnlCuda::getMaxGridSize(), tnlCuda::getNumberOfBlocks( size, cudaBlockSize.x ) );      

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

#endif /* TNLDIAGONALPRECONDITIONER_IMPL_H_ */
