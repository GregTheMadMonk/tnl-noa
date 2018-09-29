/***************************************************************************
                          VectorOperationsHost_impl.h  -  description
                             -------------------
    begin                : Nov 8, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/Math.h>
#include <TNL/Containers/Algorithms/VectorOperations.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

static const int OpenMPVectorOperationsThreshold = 512; // TODO: check this threshold
static const int PrefetchDistance = 128;

template< typename Vector >
void
VectorOperations< Devices::Host >::
addElement( Vector& v,
            const typename Vector::IndexType i,
            const typename Vector::RealType& value )
{
   v[ i ] += value;
}

template< typename Vector >
void
VectorOperations< Devices::Host >::
addElement( Vector& v,
            const typename Vector::IndexType i,
            const typename Vector::RealType& value,
            const typename Vector::RealType& thisElementMultiplicator )
{
   v[ i ] = thisElementMultiplicator * v[ i ] + value;
}

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorMax( const Vector& v )
{
   typedef typename Vector::RealType RealType;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   Algorithms::ParallelReductionMax< RealType, ResultType > operation;
   return Reduction< Devices::Host >::reduce( operation,
                                              v.getSize(),
                                              v.getData(),
                                              ( RealType* ) 0 );
}

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorMin( const Vector& v )
{
   typedef typename Vector::RealType RealType;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   Algorithms::ParallelReductionMin< RealType, ResultType > operation;
   return Reduction< Devices::Host >::reduce( operation,
                                              v.getSize(),
                                              v.getData(),
                                              ( RealType* ) 0 );
}

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorAbsMax( const Vector& v )
{
   typedef typename Vector::RealType RealType;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   Algorithms::ParallelReductionAbsMax< RealType, ResultType > operation;
   return Reduction< Devices::Host >::reduce( operation,
                                              v.getSize(),
                                              v.getData(),
                                              ( RealType* ) 0 );
}


template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorAbsMin( const Vector& v )
{
   typedef typename Vector::RealType RealType;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   Algorithms::ParallelReductionAbsMin< RealType, ResultType > operation;
   return Reduction< Devices::Host >::reduce( operation,
                                              v.getSize(),
                                              v.getData(),
                                              ( RealType* ) 0 );
}

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorL1Norm( const Vector& v )
{
   typedef typename Vector::RealType RealType;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   Algorithms::ParallelReductionAbsSum< RealType, ResultType > operation;
   return Reduction< Devices::Host >::reduce( operation,
                                              v.getSize(),
                                              v.getData(),
                                              ( RealType* ) 0 );
}

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorL2Norm( const Vector& v )
{
   typedef typename Vector::RealType Real;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   Algorithms::ParallelReductionL2Norm< Real, ResultType > operation;
   const ResultType result = Reduction< Devices::Host >::reduce( operation,
                                                                 v.getSize(),
                                                                 v.getData(),
                                                                 ( Real* ) 0 );
   return std::sqrt( result );
}

template< typename Vector, typename ResultType, typename Real_ >
ResultType
VectorOperations< Devices::Host >::
getVectorLpNorm( const Vector& v,
                 const Real_ p )
{
   typedef typename Vector::RealType Real;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_GE( p, 1.0, "Parameter of the L^p norm must be at least 1.0." );

   if( p == 1.0 )
      return getVectorL1Norm< Vector, ResultType >( v );
   if( p == 2.0 )
      return getVectorL2Norm< Vector, ResultType >( v );

   Algorithms::ParallelReductionLpNorm< Real, ResultType, Real_ > operation;
   operation.setPower( p );
   const ResultType result = Reduction< Devices::Host >::reduce( operation,
                                                                 v.getSize(),
                                                                 v.getData(),
                                                                 ( Real* ) 0 );
   return std::pow( result, 1.0 / p );
}

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorSum( const Vector& v )
{
   typedef typename Vector::RealType Real;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   Algorithms::ParallelReductionSum< Real, ResultType > operation;
   return Reduction< Devices::Host >::reduce( operation,
                                              v.getSize(),
                                              v.getData(),
                                              ( Real* ) 0 );
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorDifferenceMax( const Vector1& v1,
                        const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   Algorithms::ParallelReductionDiffMax< typename Vector1::RealType, typename Vector2::RealType, ResultType > operation;
   return Reduction< Devices::Host >::reduce( operation,
                                              v1.getSize(),
                                              v1.getData(),
                                              v2.getData() );
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorDifferenceMin( const Vector1& v1,
                        const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   Algorithms::ParallelReductionDiffMin< typename Vector1::RealType, typename Vector2::RealType, ResultType > operation;
   return Reduction< Devices::Host >::reduce( operation,
                                              v1.getSize(),
                                              v1.getData(),
                                              v2.getData() );
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorDifferenceAbsMax( const Vector1& v1,
                           const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   Algorithms::ParallelReductionDiffAbsMax< typename Vector1::RealType, typename Vector2::RealType, ResultType > operation;
   return Reduction< Devices::Host >::reduce( operation,
                                              v1.getSize(),
                                              v1.getData(),
                                              v2.getData() );
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorDifferenceAbsMin( const Vector1& v1,
                           const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   Algorithms::ParallelReductionDiffAbsMin< typename Vector1::RealType, typename Vector2::RealType, ResultType > operation;
   return Reduction< Devices::Host >::reduce( operation,
                                              v1.getSize(),
                                              v1.getData(),
                                              v2.getData() );
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorDifferenceL1Norm( const Vector1& v1,
                           const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   Algorithms::ParallelReductionDiffAbsSum< typename Vector1::RealType, typename Vector2::RealType, ResultType > operation;
   return Reduction< Devices::Host >::reduce( operation,
                                              v1.getSize(),
                                              v1.getData(),
                                              v2.getData() );
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorDifferenceL2Norm( const Vector1& v1,
                           const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   Algorithms::ParallelReductionDiffL2Norm< typename Vector1::RealType, typename Vector2::RealType, ResultType > operation;
   const ResultType result = Reduction< Devices::Host >::reduce( operation,
                                                                 v1.getSize(),
                                                                 v1.getData(),
                                                                 v2.getData() );
   return std::sqrt( result );
}


template< typename Vector1, typename Vector2, typename ResultType, typename Real_ >
ResultType
VectorOperations< Devices::Host >::
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
   const ResultType result = Reduction< Devices::Host >::reduce( operation,
                                                                 v1.getSize(),
                                                                 v1.getData(),
                                                                 v2.getData() );
   return std::pow( result, 1.0 / p );
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorDifferenceSum( const Vector1& v1,
                        const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   Algorithms::ParallelReductionDiffSum< typename Vector1::RealType, typename Vector2::RealType, ResultType > operation;
   return Reduction< Devices::Host >::reduce( operation,
                                              v1.getSize(),
                                              v1.getData(),
                                              v2.getData() );
}


template< typename Vector >
void
VectorOperations< Devices::Host >::
vectorScalarMultiplication( Vector& v,
                            const typename Vector::RealType& alpha )
{
   typedef typename Vector::IndexType Index;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   const Index n = v.getSize();
#ifdef HAVE_OPENMP
#pragma omp parallel for if( n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i ++ )
      v[ i ] *= alpha;
}


template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getScalarProduct( const Vector1& v1,
                  const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   Algorithms::ParallelReductionScalarProduct< typename Vector1::RealType, typename Vector2::RealType, ResultType > operation;
   return Reduction< Devices::Host >::reduce( operation,
                                              v1.getSize(),
                                              v1.getData(),
                                              v2.getData() );
}

template< typename Vector1, typename Vector2 >
void
VectorOperations< Devices::Host >::
addVector( Vector1& y,
           const Vector2& x,
           const typename Vector2::RealType& alpha,
           const typename Vector1::RealType& thisMultiplicator )
{
   typedef typename Vector1::IndexType Index;

   TNL_ASSERT_GT( x.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( x.getSize(), y.getSize(), "The vector sizes must be the same." );

   const Index n = y.getSize();

   if( thisMultiplicator == 1.0 )
#ifdef HAVE_OPENMP
#pragma omp parallel for if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
      for( Index i = 0; i < n; i ++ )
         y[ i ] += alpha * x[ i ];
   else
#ifdef HAVE_OPENMP
#pragma omp parallel for if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
      for( Index i = 0; i < n; i ++ )
         y[ i ] = thisMultiplicator * y[ i ] + alpha * x[ i ];
}

template< typename Vector1,
          typename Vector2,
          typename Vector3 >
void
VectorOperations< Devices::Host >::
addVectors( Vector1& v,
            const Vector2& v1,
            const typename Vector2::RealType& multiplicator1,
            const Vector3& v2,
            const typename Vector3::RealType& multiplicator2,
            const typename Vector1::RealType& thisMultiplicator )
{
   typedef typename Vector1::IndexType Index;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v.getSize(), v1.getSize(), "The vector sizes must be the same." );
   TNL_ASSERT_EQ( v.getSize(), v2.getSize(), "The vector sizes must be the same." );
 
   const Index n = v.getSize();
   if( thisMultiplicator == 1.0 )
#ifdef HAVE_OPENMP
#pragma omp parallel for if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
      for( Index i = 0; i < n; i ++ )
         v[ i ] += multiplicator1 * v1[ i ] + multiplicator2 * v2[ i ];
   else
#ifdef HAVE_OPENMP
#pragma omp parallel for if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
      for( Index i = 0; i < n; i ++ )
         v[ i ] = thisMultiplicator * v[ i ] + multiplicator1 * v1[ i ] + multiplicator2 * v2[ i ];
}

template< typename Vector >
void
VectorOperations< Devices::Host >::
computePrefixSum( Vector& v,
                  typename Vector::IndexType begin,
                  typename Vector::IndexType end )
{
   typedef typename Vector::IndexType Index;

   // TODO: parallelize with OpenMP
   for( Index i = begin + 1; i < end; i++ )
      v[ i ] += v[ i - 1 ];
}

template< typename Vector >
void
VectorOperations< Devices::Host >::
computeExclusivePrefixSum( Vector& v,
                           typename Vector::IndexType begin,
                           typename Vector::IndexType end )
{
   typedef typename Vector::IndexType Index;
   typedef typename Vector::RealType Real;

   // TODO: parallelize with OpenMP
   Real aux( v[ begin ] );
   v[ begin ] = 0.0;
   for( Index i = begin + 1; i < end; i++ )
   {
      Real x = v[ i ];
      v[ i ] = aux;
      aux += x;
   }
}

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
