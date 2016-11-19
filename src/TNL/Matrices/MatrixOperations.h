#pragma once

/*
 * TODO: This is just a temporary file, used only in the CWYGMRES solver.
 * The algorithms should be incorporated into the Matrices::Dense class.
 */

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Math.h>

namespace TNL {

template< typename DeviceType = Devices::Host >
class MatrixOperations
{
public:
   /*
    * This function performs the matrix-vector multiplication
    *    y = alpha * A * x + beta * y
    * where:
    *    alpha and beta are scalars,
    *    A is an (lda by n) matrix stored in column-major format,
    *    lda >= m is the leading dimension of two-dimensional array used to store matrix A,
    *    x is a vector of n elements,
    *    y is a vector of m elements.
    */
   template< typename RealType,
             typename IndexType >
   static void
   gemv( const IndexType& m,
         const IndexType& n,
         const RealType& alpha,
         const RealType* A,
         const IndexType& lda,
         const RealType* x,
         const RealType& beta,
         RealType* y )
   {
      Assert( m <= lda, );

      if( beta != 0.0 ) {
#ifdef HAVE_OPENMP
#pragma omp parallel for if( TNL::Devices::Host::isOMPEnabled() )
#endif
         for( IndexType j = 0; j < m; j++ ) {
            RealType tmp = 0.0;
            for( int k = 0; k < n; k++ )
               tmp += A[ j + k * lda ] * x[ k ];
            y[ j ] = alpha * tmp + beta * y[ j ];
         }
      }
      else {
         // the vector y might be uninitialized, and 0.0 * NaN = NaN
#ifdef HAVE_OPENMP
#pragma omp parallel for if( TNL::Devices::Host::isOMPEnabled() )
#endif
         for( IndexType j = 0; j < m; j++ ) {
            RealType tmp = 0.0;
            for( int k = 0; k < n; k++ )
               tmp += A[ j + k * lda ] * x[ k ];
            y[ j ] = alpha * tmp;
         }
      }
   }
};


// CUDA kernels
#ifdef HAVE_CUDA

#if (__CUDA_ARCH__ >= 300 )
   static constexpr int Gemv_minBlocksPerMultiprocessor = 8;
#else
   static constexpr int Gemv_minBlocksPerMultiprocessor = 4;
#endif

template< typename RealType,
          typename IndexType >
__global__ void
GemvCudaKernel( const IndexType m,
                const IndexType n,
                const RealType alpha,
                const RealType* A,
                const IndexType lda,
                const RealType* x,
                const RealType beta,
                RealType* y )
{
   IndexType elementIdx = blockIdx.x * blockDim.x + threadIdx.x;
   const IndexType gridSize = blockDim.x * gridDim.x;

   RealType* shx = Devices::Cuda::getSharedMemory< RealType >();

   if( threadIdx.x < n )
      shx[ threadIdx.x ] = x[ threadIdx.x ];
   __syncthreads();

   if( beta != 0.0 ) {
      while( elementIdx < m ) {
         RealType tmp = 0.0;
         for( IndexType k = 0; k < n; k++ )
            tmp += A[ elementIdx + k * lda ] * shx[ k ];
         y[ elementIdx ] = alpha * tmp + beta * y[ elementIdx ];
         elementIdx += gridSize;
      }
   }
   else {
      // the vector y might be uninitialized, and 0.0 * NaN = NaN
      while( elementIdx < m ) {
         RealType tmp = 0.0;
         for( IndexType k = 0; k < n; k++ )
            tmp += A[ elementIdx + k * lda ] * shx[ k ];
         y[ elementIdx ] = alpha * tmp;
         elementIdx += gridSize;
      }
   }
}
#endif

// specialization for CUDA
template<>
class MatrixOperations< Devices::Cuda >
{
public:
   /*
    * This function performs the matrix-vector multiplication
    *    y = alpha * A * x + beta * y
    * where:
    *    alpha and beta are scalars,
    *    A is an (lda by n) matrix stored in column-major format on Devices::Cuda,
    *    lda >= m is the leading dimension of two-dimensional array used to store matrix A,
    *    x is a vector of n elements, stored on Devices::Host,
    *    y is a vector of m elements, stored on Devices::Cuda.
    */
   template< typename RealType,
             typename IndexType >
   static void
   gemv( const IndexType& m,
         const IndexType& n,
         const RealType& alpha,
         const RealType* A,
         const IndexType& lda,
         const RealType* x,
         const RealType& beta,
         RealType* y )
   {
      Assert( m <= lda, );
      Assert( n <= 256,
              std::cerr << "The gemv kernel is optimized only for small 'n' and assumes that n <= 256." << std::endl; );

#ifdef HAVE_CUDA
      Containers::Vector< RealType, Devices::Cuda, IndexType > xDevice;
      xDevice.setSize( n );
      if( ! Containers::Algorithms::ArrayOperations< Devices::Cuda, Devices::Host >::copyMemory< RealType, RealType, IndexType >( xDevice.getData(), x, n ) )
         throw 1;

      dim3 blockSize( 256 );
      dim3 gridSize;
      const IndexType desGridSize = 4 * Gemv_minBlocksPerMultiprocessor
                                      * Devices::CudaDeviceInfo::getCudaMultiprocessors( Devices::CudaDeviceInfo::getActiveDevice() );
      gridSize.x = min( desGridSize, Devices::Cuda::getNumberOfBlocks( m, blockSize.x ) );

      GemvCudaKernel<<< gridSize, blockSize, n * sizeof( RealType ) >>>(
            m, n,
            alpha, A, lda,
            xDevice.getData(), beta, y );
      checkCudaDevice;
#else
      CudaSupportMissingMessage;
#endif
   }
};

} // namespace TNL
