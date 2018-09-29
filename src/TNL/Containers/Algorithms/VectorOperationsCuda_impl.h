/***************************************************************************
                          VectorOperationsCuda_impl.h  -  description
                             -------------------
    begin                : Nov 8, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/tnlConfig.h>
#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Containers/Algorithms/VectorOperations.h>
#include <TNL/Containers/Algorithms/cuda-prefix-sum.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename Vector >
void
VectorOperations< Devices::Cuda >::
addElement( Vector& v,
            const typename Vector::IndexType i,
            const typename Vector::RealType& value )
{
   v[ i ] += value;
}

template< typename Vector >
void
VectorOperations< Devices::Cuda >::
addElement( Vector& v,
            const typename Vector::IndexType i,
            const typename Vector::RealType& value,
            const typename Vector::RealType& thisElementMultiplicator )
{
   v[ i ] = thisElementMultiplicator * v[ i ] + value;
}

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Cuda >::
getVectorMax( const Vector& v )
{
   typedef typename Vector::RealType RealType;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   Algorithms::ParallelReductionMax< RealType, ResultType > operation;
   return Reduction< Devices::Cuda >::reduce( operation,
                                              v.getSize(),
                                              v.getData(),
                                              ( RealType* ) 0 );
}

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Cuda >::
getVectorMin( const Vector& v )
{
   typedef typename Vector::RealType RealType;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   Algorithms::ParallelReductionMin< RealType, ResultType > operation;
   return Reduction< Devices::Cuda >::reduce( operation,
                                              v.getSize(),
                                              v.getData(),
                                              ( RealType* ) 0 );
}

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Cuda >::
getVectorAbsMax( const Vector& v )
{
   typedef typename Vector::RealType RealType;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   Algorithms::ParallelReductionAbsMax< RealType, ResultType > operation;
   return Reduction< Devices::Cuda >::reduce( operation,
                                              v.getSize(),
                                              v.getData(),
                                              ( RealType* ) 0 );
}

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Cuda >::
getVectorAbsMin( const Vector& v )
{
   typedef typename Vector::RealType RealType;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   Algorithms::ParallelReductionAbsMin< RealType, ResultType > operation;
   return Reduction< Devices::Cuda >::reduce( operation,
                                              v.getSize(),
                                              v.getData(),
                                              ( RealType* ) 0 );
}

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Cuda >::
getVectorL1Norm( const Vector& v )
{
   typedef typename Vector::RealType RealType;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   Algorithms::ParallelReductionAbsSum< RealType, ResultType > operation;
   return Reduction< Devices::Cuda >::reduce( operation,
                                              v.getSize(),
                                              v.getData(),
                                              ( RealType* ) 0 );
}

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Cuda >::
getVectorL2Norm( const Vector& v )
{
   typedef typename Vector::RealType Real;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   Algorithms::ParallelReductionL2Norm< Real, ResultType > operation;
   const ResultType result = Reduction< Devices::Cuda >::reduce( operation,
                                                                 v.getSize(),
                                                                 v.getData(),
                                                                 ( Real* ) 0 );
   return std::sqrt( result );
}

template< typename Vector, typename ResultType, typename Real_ >
ResultType
VectorOperations< Devices::Cuda >::
getVectorLpNorm( const Vector& v,
                 const Real_ p )
{
   typedef typename Vector::RealType Real;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_GE( p, 1.0, "Parameter of the L^p norm must be at least 1.0." );

   if( p == 1 )
      return getVectorL1Norm< Vector, ResultType >( v );
   if( p == 2 )
      return getVectorL2Norm< Vector, ResultType >( v );

   Algorithms::ParallelReductionLpNorm< Real, ResultType, Real_ > operation;
   operation.setPower( p );
   const ResultType result = Reduction< Devices::Cuda >::reduce( operation,
                                                                 v.getSize(),
                                                                 v.getData(),
                                                                 ( Real* ) 0 );
   return std::pow( result, 1.0 / p );
}

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Cuda >::
getVectorSum( const Vector& v )
{
   typedef typename Vector::RealType Real;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   Algorithms::ParallelReductionSum< Real, ResultType > operation;
   return Reduction< Devices::Cuda >::reduce( operation,
                                              v.getSize(),
                                              v.getData(),
                                              ( Real* ) 0 );
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Cuda >::
getVectorDifferenceMax( const Vector1& v1,
                        const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   Algorithms::ParallelReductionDiffMax< typename Vector1::RealType, typename Vector2::RealType, ResultType > operation;
   return Reduction< Devices::Cuda >::reduce( operation,
                                              v1.getSize(),
                                              v1.getData(),
                                              v2.getData() );
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Cuda >::
getVectorDifferenceMin( const Vector1& v1,
                        const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   Algorithms::ParallelReductionDiffMin< typename Vector1::RealType, typename Vector2::RealType, ResultType > operation;
   return Reduction< Devices::Cuda >::reduce( operation,
                                              v1.getSize(),
                                              v1.getData(),
                                              v2.getData() );
}


template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Cuda >::
getVectorDifferenceAbsMax( const Vector1& v1,
                           const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   Algorithms::ParallelReductionDiffAbsMax< typename Vector1::RealType, typename Vector2::RealType, ResultType > operation;
   return Reduction< Devices::Cuda >::reduce( operation,
                                              v1.getSize(),
                                              v1.getData(),
                                              v2.getData() );
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Cuda >::
getVectorDifferenceAbsMin( const Vector1& v1,
                           const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   Algorithms::ParallelReductionDiffAbsMin< typename Vector1::RealType, typename Vector2::RealType, ResultType > operation;
   return Reduction< Devices::Cuda >::reduce( operation,
                                              v1.getSize(),
                                              v1.getData(),
                                              v2.getData() );
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Cuda >::
getVectorDifferenceL1Norm( const Vector1& v1,
                           const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   Algorithms::ParallelReductionDiffAbsSum< typename Vector1::RealType, typename Vector2::RealType, ResultType > operation;
   return Reduction< Devices::Cuda >::reduce( operation,
                                              v1.getSize(),
                                              v1.getData(),
                                              v2.getData() );
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Cuda >::
getVectorDifferenceL2Norm( const Vector1& v1,
                           const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   Algorithms::ParallelReductionDiffL2Norm< typename Vector1::RealType, typename Vector2::RealType, ResultType > operation;
   const ResultType result = Reduction< Devices::Cuda >::reduce( operation,
                                                                 v1.getSize(),
                                                                 v1.getData(),
                                                                 v2.getData() );
   return std::sqrt( result );
}

template< typename Vector1, typename Vector2, typename ResultType, typename Real_ >
ResultType
VectorOperations< Devices::Cuda >::
getVectorDifferenceLpNorm( const Vector1& v1,
                           const Vector2& v2,
                           const Real_ p )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );
   TNL_ASSERT_GE( p, 1.0, "Parameter of the L^p norm must be at least 1.0." );

   if( p == 1.0 )
      return getVectorDifferenceL1Norm< Vector1, Vector2, ResultType >( v1, v2 );
   if( p == 2.0 )
      return getVectorDifferenceL2Norm< Vector1, Vector2, ResultType >( v1, v2 );

   Algorithms::ParallelReductionDiffLpNorm< typename Vector1::RealType, typename Vector2::RealType, ResultType, Real_ > operation;
   operation.setPower( p );
   const ResultType result = Reduction< Devices::Cuda >::reduce( operation,
                                                                 v1.getSize(),
                                                                 v1.getData(),
                                                                 v2.getData() );
   return std::pow( result, 1.0 / p );
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Cuda >::
getVectorDifferenceSum( const Vector1& v1,
                        const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   Algorithms::ParallelReductionDiffSum< typename Vector1::RealType, typename Vector2::RealType, ResultType > operation;
   return Reduction< Devices::Cuda >::reduce( operation,
                                              v1.getSize(),
                                              v1.getData(),
                                              v2.getData() );
}

#ifdef HAVE_CUDA
template< typename Real, typename Index >
__global__ void
vectorScalarMultiplicationCudaKernel( Real* data,
                                      Index size,
                                      Real alpha )
{
   Index elementIdx = blockDim.x * blockIdx.x + threadIdx.x;
   const Index maxGridSize = blockDim.x * gridDim.x;
   while( elementIdx < size )
   {
      data[ elementIdx ] *= alpha;
      elementIdx += maxGridSize;
   }
}
#endif

template< typename Vector >
void
VectorOperations< Devices::Cuda >::
vectorScalarMultiplication( Vector& v,
                            const typename Vector::RealType& alpha )
{
   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

#ifdef HAVE_CUDA
   typedef typename Vector::IndexType Index;
   dim3 blockSize( 0 ), gridSize( 0 );
   const Index& size = v.getSize();
   blockSize.x = 256;
   Index blocksNumber = ceil( ( double ) size / ( double ) blockSize.x );
   gridSize.x = min( blocksNumber, Devices::Cuda::getMaxGridSize() );
   vectorScalarMultiplicationCudaKernel<<< gridSize, blockSize >>>( v.getData(),
                                                                    size,
                                                                    alpha );
   TNL_CHECK_CUDA_DEVICE;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}


template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Cuda >::
getScalarProduct( const Vector1& v1,
                  const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   Algorithms::ParallelReductionScalarProduct< typename Vector1::RealType, typename Vector2::RealType, ResultType > operation;
   return Reduction< Devices::Cuda >::reduce( operation,
                                              v1.getSize(),
                                              v1.getData(),
                                              v2.getData() );
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Index >
__global__ void
vectorAddVectorCudaKernel( Real* y,
                           const Real* x,
                           const Index size,
                           const Real alpha,
                           const Real thisMultiplicator )
{
   Index elementIdx = blockDim.x * blockIdx.x + threadIdx.x;
   const Index maxGridSize = blockDim.x * gridDim.x;
   if( thisMultiplicator == 1.0 )
      while( elementIdx < size )
      {
         y[ elementIdx ] += alpha * x[ elementIdx ];
         elementIdx += maxGridSize;
      }
   else
      while( elementIdx < size )
      {
         y[ elementIdx ] = thisMultiplicator * y[ elementIdx ] + alpha * x[ elementIdx ];
         elementIdx += maxGridSize;
      }
}
#endif

template< typename Vector1, typename Vector2 >
void
VectorOperations< Devices::Cuda >::
addVector( Vector1& y,
           const Vector2& x,
           const typename Vector2::RealType& alpha,
           const typename Vector1::RealType& thisMultiplicator )
{
   TNL_ASSERT_GT( x.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( x.getSize(), y.getSize(), "The vector sizes must be the same." );

#ifdef HAVE_CUDA
   typedef typename Vector1::IndexType Index;

   dim3 blockSize( 0 ), gridSize( 0 );

   const Index& size = x.getSize();
   dim3 cudaBlockSize( 256 );
   dim3 cudaBlocks;
   cudaBlocks.x = min( Devices::Cuda::getMaxGridSize(), Devices::Cuda::getNumberOfBlocks( size, cudaBlockSize.x ) );

   vectorAddVectorCudaKernel<<< cudaBlocks, cudaBlockSize >>>( y.getData(),
                                                               x.getData(),
                                                               size,
                                                               alpha,
                                                               thisMultiplicator);
   TNL_CHECK_CUDA_DEVICE;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Index >
__global__ void
vectorAddVectorsCudaKernel( Real* v,
                            const Real* v1,
                            const Real* v2,
                            const Index size,
                            const Real multiplicator1,
                            const Real multiplicator2,
                            const Real thisMultiplicator )
{
   Index elementIdx = blockDim.x * blockIdx.x + threadIdx.x;
   const Index maxGridSize = blockDim.x * gridDim.x;
   if( thisMultiplicator == 1.0 )
      while( elementIdx < size )
      {
         v[ elementIdx ] += multiplicator1 * v1[ elementIdx ] +
                            multiplicator2 * v2[ elementIdx ];
         elementIdx += maxGridSize;
      }
   else
      while( elementIdx < size )
      {
         v[ elementIdx ] = thisMultiplicator * v[ elementIdx ] +
                           multiplicator1 * v1[ elementIdx ] +
                           multiplicator2 * v2[ elementIdx ];
         elementIdx += maxGridSize;
      }
}
#endif

template< typename Vector1,
          typename Vector2,
          typename Vector3 >
void
VectorOperations< Devices::Cuda >::
addVectors( Vector1& v,
            const Vector2& v1,
            const typename Vector2::RealType& multiplicator1,
            const Vector3& v2,
            const typename Vector3::RealType& multiplicator2,
            const typename Vector1::RealType& thisMultiplicator )
{
   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v.getSize(), v1.getSize(), "The vector sizes must be the same." );
   TNL_ASSERT_EQ( v.getSize(), v2.getSize(), "The vector sizes must be the same." );

#ifdef HAVE_CUDA
   typedef typename Vector1::IndexType Index;
   dim3 blockSize( 0 ), gridSize( 0 );

   const Index& size = v.getSize();
   dim3 cudaBlockSize( 256 );
   dim3 cudaBlocks;
   cudaBlocks.x = min( Devices::Cuda::getMaxGridSize(), Devices::Cuda::getNumberOfBlocks( size, cudaBlockSize.x ) );

   vectorAddVectorsCudaKernel<<< cudaBlocks, cudaBlockSize >>>( v.getData(),
                                                                v1.getData(),
                                                                v2.getData(),
                                                                size,
                                                                multiplicator1,
                                                                multiplicator2,
                                                                thisMultiplicator);
   TNL_CHECK_CUDA_DEVICE;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename Vector >
void
VectorOperations< Devices::Cuda >::
computePrefixSum( Vector& v,
                  typename Vector::IndexType begin,
                  typename Vector::IndexType end )
{
#ifdef HAVE_CUDA
   typedef Algorithms::ParallelReductionSum< typename Vector::RealType > OperationType;

   OperationType operation;
   Algorithms::cudaPrefixSum< typename Vector::RealType,
                              OperationType,
                              typename Vector::IndexType >
                                 ( end - begin,
                                   256,
                                   &v.getData()[ begin ],
                                   &v.getData()[ begin ],
                                   operation,
                                   Algorithms::PrefixSumType::inclusive );
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename Vector >
void
VectorOperations< Devices::Cuda >::
computeExclusivePrefixSum( Vector& v,
                           typename Vector::IndexType begin,
                           typename Vector::IndexType end )
{
#ifdef HAVE_CUDA
   typedef Algorithms::ParallelReductionSum< typename Vector::RealType > OperationType;

   OperationType operation;
   Algorithms::cudaPrefixSum< typename Vector::RealType,
                              OperationType,
                              typename Vector::IndexType >
                                 ( end - begin,
                                   256,
                                   &v.getData()[ begin ],
                                   &v.getData()[ begin ],
                                   operation,
                                   Algorithms::PrefixSumType::exclusive );
#endif
}

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
