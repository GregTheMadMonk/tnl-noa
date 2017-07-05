/***************************************************************************
                          MatrixOperations.h  -  description
                             -------------------
    begin                : May 13, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

/*
 * TODO: This is just a temporary file, used only in the CWYGMRES solver.
 * The algorithms should be incorporated into the Matrices::Dense class.
 */

#include <TNL/Exceptions/CudaSupportMissing.h>
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
    *
    * It is assumed that n is much smaller than m.
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
      TNL_ASSERT_GT( m, 0, "m must be positive" );
      TNL_ASSERT_GT( n, 0, "n must be positive" );
      TNL_ASSERT_GE( lda, m, "lda must be at least m" );

      RealType alphax[ n ];
      for( IndexType k = 0; k < n; k++ )
         alphax[ k ] = alpha * x[ k ];

      if( n == 1 ) {
         if( beta != 0.0 ) {
            #ifdef HAVE_OPENMP
            #pragma omp parallel for if( TNL::Devices::Host::isOMPEnabled() )
            #endif
            for( IndexType j = 0; j < m; j++ )
               y[ j ] = A[ j ] * alphax[ 0 ] + beta * y[ j ];
         }
         else {
            // the vector y might be uninitialized, and 0.0 * NaN = NaN
            #ifdef HAVE_OPENMP
            #pragma omp parallel for if( TNL::Devices::Host::isOMPEnabled() )
            #endif
            for( IndexType j = 0; j < m; j++ )
               y[ j ] = A[ j ] * alphax[ 0 ];
         }
      }
      else {
         // the matrix A should be accessed column-wise so we split the work into small
         // blocks and each block process by columns, either parallelly or serially
         constexpr int block_size = 128;
         const int blocks = m / block_size;

         #ifdef HAVE_OPENMP
         #pragma omp parallel if( TNL::Devices::Host::isOMPEnabled() && blocks >= 2 )
         #endif
         {
            RealType aux[ block_size ];

            #ifdef HAVE_OPENMP
            #pragma omp for nowait
            #endif
            for( int b = 0; b < blocks; b++ ) {
               const int block_offset = b * block_size;

               // initialize array for thread-local results
               for( int j = 0; j < block_size; j++ )
                  aux[ j ] = 0.0;

               // compute aux = A * alphax
               for( int k = 0; k < n; k++ ) {
                  const int offset = block_offset + k * lda;
                  for( int j = 0; j < block_size; j++ )
                     aux[ j ] += A[ offset + j ] * alphax[ k ];
               }

               // write result: y = aux + beta * y
               if( beta != 0.0 ) {
                  for( int j = 0; j < block_size; j++ )
                     y[ block_offset + j ] = aux[ j ] + beta * y[ block_offset + j ];
               }
               else {
                  // the vector y might be uninitialized, and 0.0 * NaN = NaN
                  for( IndexType j = 0; j < block_size; j++ )
                     y[ block_offset + j ] = aux[ j ];
               }
            }

            // the first thread that reaches here processes the last, incomplete block
            #ifdef HAVE_OPENMP
            #pragma omp single nowait
            #endif
            {
               // TODO: unlike the complete blocks, the tail is traversed row-wise
               if( beta != 0.0 ) {
                  for( IndexType j = blocks * block_size; j < m; j++ ) {
                     RealType tmp = 0.0;
                     for( int k = 0; k < n; k++ )
                        tmp += A[ j + k * lda ] * alphax[ k ];
                     y[ j ] = tmp + beta * y[ j ];
                  }
               }
               else {
                  // the vector y might be uninitialized, and 0.0 * NaN = NaN
                  for( IndexType j = blocks * block_size; j < m; j++ ) {
                     RealType tmp = 0.0;
                     for( int k = 0; k < n; k++ )
                        tmp += A[ j + k * lda ] * alphax[ k ];
                     y[ j ] = tmp;
                  }
               }
            }
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
      shx[ threadIdx.x ] = alpha * x[ threadIdx.x ];
   __syncthreads();

   if( beta != 0.0 ) {
      while( elementIdx < m ) {
         RealType tmp = 0.0;
         for( IndexType k = 0; k < n; k++ )
            tmp += A[ elementIdx + k * lda ] * shx[ k ];
         y[ elementIdx ] = tmp + beta * y[ elementIdx ];
         elementIdx += gridSize;
      }
   }
   else {
      // the vector y might be uninitialized, and 0.0 * NaN = NaN
      while( elementIdx < m ) {
         RealType tmp = 0.0;
         for( IndexType k = 0; k < n; k++ )
            tmp += A[ elementIdx + k * lda ] * shx[ k ];
         y[ elementIdx ] = tmp;
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
    *
    * It is assumed that n is much smaller than m.
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
      TNL_ASSERT( m <= lda, );
      TNL_ASSERT( n <= 256,
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
      TNL_CHECK_CUDA_DEVICE;
#else
      throw Exceptions::CudaSupportMissing();
#endif
   }
};

} // namespace TNL
