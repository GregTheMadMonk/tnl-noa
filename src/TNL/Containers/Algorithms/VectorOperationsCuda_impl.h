/***************************************************************************
                          VectorOperationsCuda_impl.h  -  description
                             -------------------
    begin                : Nov 8, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

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

template< typename Vector, typename Scalar >
void
VectorOperations< Devices::Cuda >::
addElement( Vector& v,
            const typename Vector::IndexType i,
            const typename Vector::RealType& value,
            const Scalar thisElementMultiplicator )
{
   v[ i ] = thisElementMultiplicator * v[ i ] + value;
}

/*template< typename Vector, typename ResultType >
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

template< typename Vector, typename ResultType, typename Scalar >
ResultType
VectorOperations< Devices::Cuda >::
getVectorLpNorm( const Vector& v,
                 const Scalar p )
{
   typedef typename Vector::RealType Real;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_GE( p, 1.0, "Parameter of the L^p norm must be at least 1.0." );

   if( p == 1 )
      return getVectorL1Norm< Vector, ResultType >( v );
   if( p == 2 )
      return getVectorL2Norm< Vector, ResultType >( v );

   Algorithms::ParallelReductionLpNorm< Real, ResultType, Scalar > operation;
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

template< typename Vector1, typename Vector2, typename ResultType, typename Scalar >
ResultType
VectorOperations< Devices::Cuda >::
getVectorDifferenceLpNorm( const Vector1& v1,
                           const Vector2& v2,
                           const Scalar p )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );
   TNL_ASSERT_GE( p, 1.0, "Parameter of the L^p norm must be at least 1.0." );

   if( p == 1.0 )
      return getVectorDifferenceL1Norm< Vector1, Vector2, ResultType >( v1, v2 );
   if( p == 2.0 )
      return getVectorDifferenceL2Norm< Vector1, Vector2, ResultType >( v1, v2 );

   Algorithms::ParallelReductionDiffLpNorm< typename Vector1::RealType, typename Vector2::RealType, ResultType, Scalar > operation;
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
}*/

#ifdef HAVE_CUDA
template< typename Real, typename Index, typename Scalar >
__global__ void
vectorScalarMultiplicationCudaKernel( Real* data,
                                      Index size,
                                      Scalar alpha )
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

template< typename Vector, typename Scalar >
void
VectorOperations< Devices::Cuda >::
vectorScalarMultiplication( Vector& v,
                            const Scalar alpha )
{
   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

#ifdef HAVE_CUDA
   typedef typename Vector::IndexType Index;
   dim3 blockSize( 0 ), gridSize( 0 );
   const Index& size = v.getSize();
   blockSize.x = 256;
   Index blocksNumber = TNL::ceil( ( double ) size / ( double ) blockSize.x );
   gridSize.x = min( blocksNumber, Devices::Cuda::getMaxGridSize() );
   vectorScalarMultiplicationCudaKernel<<< gridSize, blockSize >>>( v.getData(),
                                                                    size,
                                                                    alpha );
   TNL_CHECK_CUDA_DEVICE;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}


/*template< typename Vector1, typename Vector2, typename ResultType >
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
}*/

#ifdef HAVE_CUDA
template< typename Real1, typename Real2, typename Index, typename Scalar1, typename Scalar2 >
__global__ void
vectorAddVectorCudaKernel( Real1* y,
                           const Real2* x,
                           const Index size,
                           const Scalar1 alpha,
                           const Scalar2 thisMultiplicator )
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

template< typename Vector1, typename Vector2, typename Scalar1, typename Scalar2 >
void
VectorOperations< Devices::Cuda >::
addVector( Vector1& _y,
           const Vector2& x,
           const Scalar1 alpha,
           const Scalar2 thisMultiplicator )
{
   TNL_ASSERT_GT( x.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( x.getSize(), _y.getSize(), "The vector sizes must be the same." );

#ifdef HAVE_CUDA
   using IndexType = typename Vector1::IndexType;
   using RealType = typename Vector1::RealType;

   RealType* y = _y.getData();
   auto add1 = [=] __cuda_callable__ ( IndexType i ) { y[ i ] += alpha * x[ i ]; };
   auto add2 = [=] __cuda_callable__ ( IndexType i ) { y[ i ] = thisMultiplicator * y[ i ] + alpha * x[ i ]; };

   if( thisMultiplicator == 1.0 )
      ParallelFor< Devices::Cuda >::exec( ( IndexType ) 0, _y.getSize(), add1 );
   else
      ParallelFor< Devices::Cuda >::exec( ( IndexType ) 0, _y.getSize(), add2 );
   /*const Index& size = x.getSize();
   dim3 cudaBlockSize( 256 );
   dim3 cudaBlocks;
   cudaBlocks.x = TNL::min( Devices::Cuda::getMaxGridSize(), Devices::Cuda::getNumberOfBlocks( size, cudaBlockSize.x ) );

   vectorAddVectorCudaKernel<<< cudaBlocks, cudaBlockSize >>>( y.getData(),
                                                               x.getData(),
                                                               size,
                                                               alpha,
                                                               thisMultiplicator);*/
   TNL_CHECK_CUDA_DEVICE;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

#ifdef HAVE_CUDA
template< typename Real1, typename Real2, typename Real3, typename Index,
          typename Scalar1, typename Scalar2, typename Scalar3 >
__global__ void
vectorAddVectorsCudaKernel( Real1* v,
                            const Real2* v1,
                            const Real3* v2,
                            const Index size,
                            const Scalar1 multiplicator1,
                            const Scalar2 multiplicator2,
                            const Scalar3 thisMultiplicator )
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

template< typename Vector1, typename Vector2, typename Vector3,
          typename Scalar1, typename Scalar2, typename Scalar3 >
void
VectorOperations< Devices::Cuda >::
addVectors( Vector1& v,
            const Vector2& v1,
            const Scalar1 multiplicator1,
            const Vector3& v2,
            const Scalar2 multiplicator2,
            const Scalar3 thisMultiplicator )
{
   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v.getSize(), v1.getSize(), "The vector sizes must be the same." );
   TNL_ASSERT_EQ( v.getSize(), v2.getSize(), "The vector sizes must be the same." );

#ifdef HAVE_CUDA
   typedef typename Vector1::IndexType Index;

   const Index& size = v.getSize();
   dim3 cudaBlockSize( 256 );
   dim3 cudaBlocks;
   cudaBlocks.x = TNL::min( Devices::Cuda::getMaxGridSize(), Devices::Cuda::getNumberOfBlocks( size, cudaBlockSize.x ) );

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
