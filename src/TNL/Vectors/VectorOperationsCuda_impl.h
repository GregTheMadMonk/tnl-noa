/***************************************************************************
                          tnlVectorOperationsCuda_impl.h  -  description
                             -------------------
    begin                : Nov 8, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/tnlConfig.h>
#include <TNL/core/cuda/cuda-prefix-sum.h>
#include <TNL/core/cuda/tnlCublasWrapper.h>

namespace TNL {
namespace Vectors {   

template< typename Vector >
void tnlVectorOperations< tnlCuda >::addElement( Vector& v,
                                                 const typename Vector::IndexType i,
                                                 const typename Vector::RealType& value )
{
   v[ i ] += value;
}

template< typename Vector >
void tnlVectorOperations< tnlCuda >::addElement( Vector& v,
                                                 const typename Vector::IndexType i,
                                                 const typename Vector::RealType& value,
                                                 const typename Vector::RealType& thisElementMultiplicator )
{
   v[ i ] = thisElementMultiplicator * v[ i ] + value;
}

template< typename Vector >
typename Vector :: RealType tnlVectorOperations< tnlCuda > :: getVectorMax( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;

   Assert( v. getSize() > 0, );

   Real result( 0 );
   tnlParallelReductionMax< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v. getSize(),
                          v. getData(),
                          ( Real* ) 0,
                          result );
   return result;
}

template< typename Vector >
typename Vector :: RealType tnlVectorOperations< tnlCuda > :: getVectorMin( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;

   Assert( v. getSize() > 0, );

   Real result( 0 );
   tnlParallelReductionMin< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v. getSize(),
                          v. getData(),
                          ( Real* ) 0,
                          result );
   return result;
}

template< typename Vector >
typename Vector :: RealType tnlVectorOperations< tnlCuda > :: getVectorAbsMax( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;

   Assert( v. getSize() > 0, );

   Real result( 0 );
   tnlParallelReductionAbsMax< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v. getSize(),
                          v. getData(),
                          ( Real* ) 0,
                          result );
   return result;
}

template< typename Vector >
typename Vector :: RealType tnlVectorOperations< tnlCuda > :: getVectorAbsMin( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;

   Assert( v. getSize() > 0, );

   Real result( 0 );
   tnlParallelReductionAbsMin< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v. getSize(),
                          v. getData(),
                          ( Real* ) 0,
                          result );
   return result;
}

template< typename Vector >
typename Vector::RealType
tnlVectorOperations< tnlCuda >::
getVectorL1Norm( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;

   Assert( v. getSize() > 0, );

   Real result( 0 );
   tnlParallelReductionAbsSum< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v. getSize(),
                          v. getData(),
                          ( Real* ) 0,
                          result );
   return result;
}

template< typename Vector >
typename Vector::RealType
tnlVectorOperations< tnlCuda >::
getVectorL2Norm( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;

   Assert( v. getSize() > 0, );

   Real result( 0 );
   tnlParallelReductionL2Norm< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v. getSize(),
                          v. getData(),
                          ( Real* ) 0,
                          result );
   return std::sqrt( result );
}


template< typename Vector >
typename Vector::RealType
tnlVectorOperations< tnlCuda >::
getVectorLpNorm( const Vector& v,
                 const typename Vector::RealType& p )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;

   Assert( v. getSize() > 0, );
   Assert( p > 0.0,
              std::cerr << " p = " << p );
 
   if( p == 1 )
      return getVectorL1Norm( v );
   if( p == 2 )
      return getVectorL2Norm( v );
   Real result( 0 );
   tnlParallelReductionLpNorm< Real, Index > operation;
   operation. setPower( p );
   reductionOnCudaDevice( operation,
                          v. getSize(),
                          v. getData(),
                          ( Real* ) 0,
                          result );
   return std::pow( result, 1.0 / p );
}

template< typename Vector >
typename Vector :: RealType tnlVectorOperations< tnlCuda > :: getVectorSum( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;

   Assert( v. getSize() > 0, );

   Real result( 0 );
   tnlParallelReductionSum< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v. getSize(),
                          v. getData(),
                          ( Real* ) 0,
                          result );
   return result;
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType tnlVectorOperations< tnlCuda > :: getVectorDifferenceMax( const Vector1& v1,
                                                            const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( v1. getSize() > 0, );
   Assert( v1. getSize() == v2. getSize(), );

   Real result( 0 );
   tnlParallelReductionDiffMax< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v1. getSize(),
                          v1. getData(),
                          v2. getData(),
                          result );
   return result;
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType tnlVectorOperations< tnlCuda > :: getVectorDifferenceMin( const Vector1& v1,
                                                            const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( v1. getSize() > 0, );
   Assert( v1. getSize() == v2. getSize(), );

   Real result( 0 );
   tnlParallelReductionDiffMin< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v1. getSize(),
                          v1. getData(),
                          v2. getData(),
                          result );
   return result;
}


template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType tnlVectorOperations< tnlCuda > :: getVectorDifferenceAbsMax( const Vector1& v1,
                                                               const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( v1. getSize() > 0, );
   Assert( v1. getSize() == v2. getSize(), );

   Real result( 0 );
   tnlParallelReductionDiffAbsMax< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v1. getSize(),
                          v1. getData(),
                          v2. getData(),
                          result );
   return result;
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType tnlVectorOperations< tnlCuda > :: getVectorDifferenceAbsMin( const Vector1& v1,
                                                            const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( v1. getSize() > 0, );
   Assert( v1. getSize() == v2. getSize(), );

   Real result( 0 );
   tnlParallelReductionDiffAbsMin< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v1. getSize(),
                          v1. getData(),
                          v2. getData(),
                          result );
   return result;
}

template< typename Vector1, typename Vector2 >
typename Vector1::RealType
tnlVectorOperations< tnlCuda >::
getVectorDifferenceL1Norm( const Vector1& v1,
                           const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( v1. getSize() > 0, );
   Assert( v1. getSize() == v2. getSize(), );

   Real result( 0 );
   tnlParallelReductionDiffAbsSum< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v1. getSize(),
                          v1. getData(),
                          v2. getData(),
                          result );
   return result;
}

template< typename Vector1, typename Vector2 >
typename Vector1::RealType
tnlVectorOperations< tnlCuda >::
getVectorDifferenceL2Norm( const Vector1& v1,
                           const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( v1. getSize() > 0, );
   Assert( v1. getSize() == v2. getSize(), );

   Real result( 0 );
   tnlParallelReductionDiffL2Norm< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v1. getSize(),
                          v1. getData(),
                          v2. getData(),
                          result );
   return ::sqrt( result );
}


template< typename Vector1, typename Vector2 >
typename Vector1::RealType
tnlVectorOperations< tnlCuda >::
getVectorDifferenceLpNorm( const Vector1& v1,
                           const Vector2& v2,
                           const typename Vector1 :: RealType& p )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( p > 0.0,
              std::cerr << " p = " << p );
   Assert( v1. getSize() > 0, );
   Assert( v1. getSize() == v2. getSize(), );

   Real result( 0 );
   tnlParallelReductionDiffLpNorm< Real, Index > operation;
   operation.setPower( p );
   reductionOnCudaDevice( operation,
                          v1. getSize(),
                          v1. getData(),
                          v2. getData(),
                          result );
   return ::pow( result, 1.0 / p );
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType tnlVectorOperations< tnlCuda > :: getVectorDifferenceSum( const Vector1& v1,
                                                         const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( v1. getSize() > 0, );
   Assert( v1. getSize() == v2. getSize(), );

   Real result( 0 );
   tnlParallelReductionDiffSum< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v1. getSize(),
                          v1. getData(),
                          v2. getData(),
                          result );
   return result;
}

#ifdef HAVE_CUDA
template< typename Real, typename Index >
__global__ void vectorScalarMultiplicationCudaKernel( Real* data,
                                                      Index size,
                                                      Real alpha )
{
   Index elementIdx = blockDim. x * blockIdx. x + threadIdx. x;
   const Index maxGridSize = blockDim. x * gridDim. x;
   while( elementIdx < size )
   {
      data[ elementIdx ] *= alpha;
      elementIdx += maxGridSize;
   }
}
#endif

template< typename Vector >
void tnlVectorOperations< tnlCuda > :: vectorScalarMultiplication( Vector& v,
                                                                   const typename Vector::RealType& alpha )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;

   Assert( v. getSize() > 0, );

   #ifdef HAVE_CUDA
      dim3 blockSize( 0 ), gridSize( 0 );
      const Index& size = v.getSize();
      blockSize. x = 256;
      Index blocksNumber = ceil( ( double ) size / ( double ) blockSize. x );
      gridSize. x = min( blocksNumber, tnlCuda::getMaxGridSize() );
      vectorScalarMultiplicationCudaKernel<<< gridSize, blockSize >>>( v.getData(),
                                                                       size,
                                                                       alpha );
      checkCudaDevice;
   #else
      tnlCudaSupportMissingMessage;;
   #endif
}


template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType tnlVectorOperations< tnlCuda > :: getScalarProduct( const Vector1& v1,
                                                                                 const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( v1. getSize() > 0, );
   Assert( v1. getSize() == v2. getSize(), );

   Real result( 0 );
/*#if defined HAVE_CUBLAS && defined HAVE_CUDA
   if( tnlCublasWrapper< typename Vector1::RealType,
                         typename Vector2::RealType,
                         typename Vector1::IndexType >::dot( v1.getData(), v1.getData(), v1.getSize(), result ) )
       return result;
#endif*/
   tnlParallelReductionScalarProduct< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v1. getSize(),
                          v1. getData(),
                          v2. getData(),
                          result );
   return result;
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Index >
__global__ void vectorAddVectorCudaKernel( Real* y,
                                           const Real* x,
                                           const Index size,
                                           const Real alpha,
                                           const Real thisMultiplicator )
{
   Index elementIdx = blockDim. x * blockIdx. x + threadIdx. x;
   const Index maxGridSize = blockDim. x * gridDim. x;
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
void tnlVectorOperations< tnlCuda > :: addVector( Vector1& y,
                                                  const Vector2& x,
                                                  const typename Vector2::RealType& alpha,
                                                  const typename Vector1::RealType& thisMultiplicator )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( y. getSize() > 0, );
   Assert( y. getSize() == x. getSize(), );
   Assert( y.getData() != 0, );
   Assert( x.getData() != 0, );


   #ifdef HAVE_CUDA
      dim3 blockSize( 0 ), gridSize( 0 );

      const Index& size = x.getSize();
      dim3 cudaBlockSize( 256 );
      dim3 cudaBlocks;
      cudaBlocks.x = min( tnlCuda::getMaxGridSize(), tnlCuda::getNumberOfBlocks( size, cudaBlockSize.x ) );

      vectorAddVectorCudaKernel<<< cudaBlocks, cudaBlockSize >>>( y.getData(),
                                                                  x.getData(),
                                                                  size,
                                                                  alpha,
                                                                  thisMultiplicator);
      checkCudaDevice;
   #else
      tnlCudaSupportMissingMessage;;
   #endif
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Index >
__global__ void vectorAddVectorsCudaKernel( Real* v,
                                            const Real* v1,
                                            const Real* v2,
                                            const Index size,
                                            const Real multiplicator1,
                                            const Real multiplicator2,
                                            const Real thisMultiplicator )
{
   Index elementIdx = blockDim. x * blockIdx. x + threadIdx. x;
   const Index maxGridSize = blockDim. x * gridDim. x;
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
tnlVectorOperations< tnlCuda >::
addVectors( Vector1& v,
            const Vector2& v1,
            const typename Vector2::RealType& multiplicator1,
            const Vector3& v2,
            const typename Vector3::RealType& multiplicator2,
            const typename Vector1::RealType& thisMultiplicator )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( v.getSize() > 0, );
   Assert( v.getSize() == v1.getSize(), );
   Assert( v.getSize() == v2.getSize(), );
   Assert( v.getData() != 0, );
   Assert( v1.getData() != 0, );
   Assert( v2.getData() != 0, );

   #ifdef HAVE_CUDA
      dim3 blockSize( 0 ), gridSize( 0 );

      const Index& size = v.getSize();
      dim3 cudaBlockSize( 256 );
      dim3 cudaBlocks;
      cudaBlocks.x = min( tnlCuda::getMaxGridSize(), tnlCuda::getNumberOfBlocks( size, cudaBlockSize.x ) );

      vectorAddVectorsCudaKernel<<< cudaBlocks, cudaBlockSize >>>( v.getData(),
                                                                   v1.getData(),
                                                                   v2.getData(),
                                                                   size,
                                                                   multiplicator1,
                                                                   multiplicator2,
                                                                   thisMultiplicator);
      checkCudaDevice;
   #else
      tnlCudaSupportMissingMessage;;
   #endif


}

template< typename Vector >
void tnlVectorOperations< tnlCuda >::computePrefixSum( Vector& v,
                                                       typename Vector::IndexType begin,
                                                       typename Vector::IndexType end )
{
   #ifdef HAVE_CUDA
   typedef tnlParallelReductionSum< typename Vector::RealType,
                                    typename Vector::IndexType > OperationType;

   OperationType operation;
   cudaPrefixSum< typename Vector::RealType,
                  OperationType,
                  typename Vector::IndexType >( end - begin,
                                                256,
                                                &v.getData()[ begin ],
                                                &v.getData()[ begin ],
                                                operation,
                                                inclusivePrefixSum );
   #else
      tnlCudaSupportMissingMessage;;
   #endif
}

template< typename Vector >
void tnlVectorOperations< tnlCuda >::computeExclusivePrefixSum( Vector& v,
                                                                typename Vector::IndexType begin,
                                                                typename Vector::IndexType end )
{
#ifdef HAVE_CUDA
   typedef tnlParallelReductionSum< typename Vector::RealType,
                                    typename Vector::IndexType > OperationType;

   OperationType operation;

   cudaPrefixSum< typename Vector::RealType,
                  OperationType,
                  typename Vector::IndexType >( end - begin,
                                                256,
                                                &v.getData()[ begin ],
                                                &v.getData()[ begin ],
                                                operation,
                                                exclusivePrefixSum );
#endif
}

} // namespace Vectors
} // namespace TNL

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#include <TNL/Vectors/Vector.h>

namespace TNL {
namespace Vectors {

/****
 * Max
 */
extern template int         tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< int, tnlCuda, int >& v );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< long int, tnlCuda, int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< float, tnlCuda, int >& v );
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< double, tnlCuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< long double, tnlCuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template int         tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< int, tnlCuda, long int >& v );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< long int, tnlCuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< float, tnlCuda, long int >& v );
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< double, tnlCuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< long double, tnlCuda, long int >& v );
#endif
#endif

/****
 * Min
 */
extern template int         tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< int, tnlCuda, int >& v );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< long int, tnlCuda, int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< float, tnlCuda, int >& v );
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< double, tnlCuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< long double, tnlCuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template int         tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< int, tnlCuda, long int >& v );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< long int, tnlCuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< float, tnlCuda, long int >& v );
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< double, tnlCuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< long double, tnlCuda, long int >& v );
#endif
#endif

/****
 * Abs max
 */
extern template int         tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< int, tnlCuda, int >& v );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< long int, tnlCuda, int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< float, tnlCuda, int >& v );
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< double, tnlCuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< long double, tnlCuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template int         tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< int, tnlCuda, long int >& v );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< long int, tnlCuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< float, tnlCuda, long int >& v );
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< double, tnlCuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< long double, tnlCuda, long int >& v );
#endif
#endif

/****
 * Abs min
 */
extern template int         tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< int, tnlCuda, int >& v );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< long int, tnlCuda, int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< float, tnlCuda, int >& v );
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< double, tnlCuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< long double, tnlCuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template int         tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< int, tnlCuda, long int >& v );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< long int, tnlCuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< float, tnlCuda, long int >& v );
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< double, tnlCuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< long double, tnlCuda, long int >& v );
#endif
#endif

/****
 * Lp norm
 */
extern template int         tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< int, tnlCuda, int >& v, const int& p );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< long int, tnlCuda, int >& v, const long int& p );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< float, tnlCuda, int >& v, const float& p );
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< double, tnlCuda, int >& v, const double& p );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< long double, tnlCuda, int >& v, const long double& p );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template int         tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< int, tnlCuda, long int >& v, const int& p );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< long int, tnlCuda, long int >& v, const long int& p );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< float, tnlCuda, long int >& v, const float& p );
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< double, tnlCuda, long int >& v, const double& p );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< long double, tnlCuda, long int >& v, const long double& p );
#endif
#endif

/****
 * Sum
 */
extern template int         tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< int, tnlCuda, int >& v );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< long int, tnlCuda, int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< float, tnlCuda, int >& v );
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< double, tnlCuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< long double, tnlCuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template int         tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< int, tnlCuda, long int >& v );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< long int, tnlCuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< float, tnlCuda, long int >& v );
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< double, tnlCuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< long double, tnlCuda, long int >& v );
#endif
#endif

/****
 * Difference max
 */
extern template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< int, tnlCuda, int >& v1, const tnlVector< int, tnlCuda, int >& v2 );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< long int, tnlCuda, int >& v1, const tnlVector< long int, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< float, tnlCuda, int >& v1,  const tnlVector< float, tnlCuda, int >& v2);
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< double, tnlCuda, int >& v1, const tnlVector< double, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< long double, tnlCuda, int >& v1, const tnlVector< long double, tnlCuda, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< int, tnlCuda, long int >& v1, const tnlVector< int, tnlCuda, long int >& v2 );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< long int, tnlCuda, long int >& v1, const tnlVector< long int, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< float, tnlCuda, long int >& v1, const tnlVector< float, tnlCuda, long int >& v2 );
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< double, tnlCuda, long int >& v1, const tnlVector< double, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< long double, tnlCuda, long int >& v1, const tnlVector< long double, tnlCuda, long int >& v2 );
#endif
#endif

/****
 * Difference min
 */
extern template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< int, tnlCuda, int >& v1, const tnlVector< int, tnlCuda, int >& v2 );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< long int, tnlCuda, int >& v1, const tnlVector< long int, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< float, tnlCuda, int >& v1,  const tnlVector< float, tnlCuda, int >& v2);
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< double, tnlCuda, int >& v1, const tnlVector< double, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< long double, tnlCuda, int >& v1, const tnlVector< long double, tnlCuda, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< int, tnlCuda, long int >& v1, const tnlVector< int, tnlCuda, long int >& v2 );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< long int, tnlCuda, long int >& v1, const tnlVector< long int, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< float, tnlCuda, long int >& v1, const tnlVector< float, tnlCuda, long int >& v2 );
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< double, tnlCuda, long int >& v1, const tnlVector< double, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< long double, tnlCuda, long int >& v1, const tnlVector< long double, tnlCuda, long int >& v2 );
#endif
#endif

/****
 * Difference abs max
 */
extern template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< int, tnlCuda, int >& v1, const tnlVector< int, tnlCuda, int >& v2 );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< long int, tnlCuda, int >& v1, const tnlVector< long int, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< float, tnlCuda, int >& v1,  const tnlVector< float, tnlCuda, int >& v2);
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< double, tnlCuda, int >& v1, const tnlVector< double, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< long double, tnlCuda, int >& v1, const tnlVector< long double, tnlCuda, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< int, tnlCuda, long int >& v1, const tnlVector< int, tnlCuda, long int >& v2 );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< long int, tnlCuda, long int >& v1, const tnlVector< long int, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< float, tnlCuda, long int >& v1, const tnlVector< float, tnlCuda, long int >& v2 );
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< double, tnlCuda, long int >& v1, const tnlVector< double, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< long double, tnlCuda, long int >& v1, const tnlVector< long double, tnlCuda, long int >& v2 );
#endif
#endif

/****
 * Difference abs min
 */
extern template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< int, tnlCuda, int >& v1, const tnlVector< int, tnlCuda, int >& v2 );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< long int, tnlCuda, int >& v1, const tnlVector< long int, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< float, tnlCuda, int >& v1,  const tnlVector< float, tnlCuda, int >& v2);
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< double, tnlCuda, int >& v1, const tnlVector< double, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< long double, tnlCuda, int >& v1, const tnlVector< long double, tnlCuda, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< int, tnlCuda, long int >& v1, const tnlVector< int, tnlCuda, long int >& v2 );
extern template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< long int, tnlCuda, long int >& v1, const tnlVector< long int, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
extern template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< float, tnlCuda, long int >& v1, const tnlVector< float, tnlCuda, long int >& v2 );
#endif
extern template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< double, tnlCuda, long int >& v1, const tnlVector< double, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< long double, tnlCuda, long int >& v1, const tnlVector< long double, tnlCuda, long int >& v2 );
#endif
#endif

} // namespace Vectors
} // namespace TNL

#endif
