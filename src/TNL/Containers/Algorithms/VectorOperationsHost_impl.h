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
   typedef typename Vector::IndexType Index;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   ResultType result = v.getElement( 0 );
   const Index n = v.getSize();
#if defined( HAVE_OPENMP ) && _OPENMP >= 201107  // OpenMP 3.1 added support for min/max reduction operations
#pragma omp parallel for reduction(max:result) if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 1; i < n; i ++ )
      result = max( result, v.getElement( i ) );
   return result;
}

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorMin( const Vector& v )
{
   typedef typename Vector::IndexType Index;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   ResultType result = v.getElement( 0 );
   const Index n = v.getSize();
#if defined( HAVE_OPENMP ) && _OPENMP >= 201107  // OpenMP 3.1 added support for min/max reduction operations
#pragma omp parallel for reduction(min:result) if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 1; i < n; i ++ )
      result = min( result, v.getElement( i ) );
   return result;
}

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorAbsMax( const Vector& v )
{
   typedef typename Vector::IndexType Index;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   ResultType result = TNL::abs( v.getElement( 0 ) );
   const Index n = v.getSize();
#if defined( HAVE_OPENMP ) && _OPENMP >= 201107  // OpenMP 3.1 added support for min/max reduction operations
#pragma omp parallel for reduction(max:result) if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 1; i < n; i ++ )
      result = max( result, (ResultType) TNL::abs( v.getElement( i ) ) );
   return result;
}


template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorAbsMin( const Vector& v )
{
   typedef typename Vector::IndexType Index;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   ResultType result = TNL::abs( v.getElement( 0 ) );
   const Index n = v.getSize();
#if defined( HAVE_OPENMP ) && _OPENMP >= 201107  // OpenMP 3.1 added support for min/max reduction operations
#pragma omp parallel for reduction(min:result) if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 1; i < n; i ++ )
      result = min( result, (ResultType) TNL::abs( v.getElement( i ) ) );
   return result;
}

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorL1Norm( const Vector& v )
{
   typedef typename Vector::IndexType Index;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   ResultType result( 0.0 );
   const Index n = v.getSize();
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:result) if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i ++ )
      result += TNL::abs( v[ i ] );
   return result;
}

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorL2Norm( const Vector& v )
{
   typedef typename Vector::RealType Real;
   typedef typename Vector::IndexType Index;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   const Index n = v.getSize();

#ifdef OPTIMIZED_VECTOR_HOST_OPERATIONS
#ifdef __GNUC__
   // We need to get the address of the first element to avoid
   // bounds checking in TNL::Array::operator[]
   const Real* V = v.getData();
#endif

   ResultType result1 = 0, result2 = 0, result3 = 0, result4 = 0;
   Index i = 0;
   const Index unroll_limit = n - n % 4;
#ifdef HAVE_OPENMP
#pragma omp parallel for \
       if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) \
       reduction(+:result1,result2,result3,result4) \
       lastprivate(i)
#endif
   for( i = 0; i < unroll_limit; i += 4 )
   {
#ifdef __GNUC__
      __builtin_prefetch(V + i + PrefetchDistance, 0, 0);
#endif
      result1 += v[ i ] * v[ i ];
      result2 += v[ i + 1 ] * v[ i + 1 ];
      result3 += v[ i + 2 ] * v[ i + 2 ];
      result4 += v[ i + 3 ] * v[ i + 3 ];
   }

   while( i < n )
   {
      result1 += v[ i ] * v[ i ];
      i++;
   }

   return std::sqrt(result1 + result2 + result3 + result4);

#else // OPTIMIZED_VECTOR_HOST_OPERATIONS

   ResultType result( 0.0 );
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:result) if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i ++ )
   {
      const Real& aux = v[ i ];
      result += aux * aux;
   }
   return std::sqrt( result );
#endif // OPTIMIZED_VECTOR_HOST_OPERATIONS
}

template< typename Vector, typename ResultType, typename Real_ >
ResultType
VectorOperations< Devices::Host >::
getVectorLpNorm( const Vector& v,
                 const Real_ p )
{
   typedef typename Vector::IndexType Index;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_GE( p, 1.0, "Parameter of the L^p norm must be at least 1.0." );

   if( p == 1.0 )
      return getVectorL1Norm< Vector, ResultType >( v );
   if( p == 2.0 )
      return getVectorL2Norm< Vector, ResultType >( v );

   ResultType result( 0.0 );
   const Index n = v.getSize();
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:result) if( TNL::Devices::Host::isOMPEnabled() &&n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i ++ )
      result += std::pow( TNL::abs( v[ i ] ), p );
   return std::pow( result, 1.0 / p );
}

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorSum( const Vector& v )
{
   typedef typename Vector::IndexType Index;

   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   ResultType result( 0.0 );
   const Index n = v.getSize();
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:result) if( TNL::Devices::Host::isOMPEnabled() &&n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i ++ )
      result += v[ i ];
   return result;
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorDifferenceMax( const Vector1& v1,
                        const Vector2& v2 )
{
   typedef typename Vector1::IndexType Index;

   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   ResultType result = v1.getElement( 0 ) - v2.getElement( 0 );
   const Index n = v1.getSize();
#if defined( HAVE_OPENMP ) && _OPENMP >= 201107  // OpenMP 3.1 added support for min/max reduction operations
#pragma omp parallel for reduction(max:result) if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 1; i < n; i ++ )
      result =  max( result, v1.getElement( i ) - v2.getElement( i ) );
   return result;
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorDifferenceMin( const Vector1& v1,
                        const Vector2& v2 )
{
   typedef typename Vector1::IndexType Index;

   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   ResultType result = v1.getElement( 0 ) - v2.getElement( 0 );
   const Index n = v1.getSize();
#if defined( HAVE_OPENMP ) && _OPENMP >= 201107  // OpenMP 3.1 added support for min/max reduction operations
#pragma omp parallel for reduction(min:result) if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 1; i < n; i ++ )
      result =  min( result, v1.getElement( i ) - v2.getElement( i ) );
   return result;
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorDifferenceAbsMax( const Vector1& v1,
                           const Vector2& v2 )
{
   typedef typename Vector1::IndexType Index;

   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   ResultType result = TNL::abs( v1.getElement( 0 ) - v2.getElement( 0 ) );
   const Index n = v1.getSize();
#if defined( HAVE_OPENMP ) && _OPENMP >= 201107  // OpenMP 3.1 added support for min/max reduction operations
#pragma omp parallel for reduction(max:result) if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 1; i < n; i ++ )
      result =  max( result, (ResultType) TNL::abs( v1.getElement( i ) - v2.getElement( i ) ) );
   return result;
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorDifferenceAbsMin( const Vector1& v1,
                           const Vector2& v2 )
{
   typedef typename Vector1::IndexType Index;

   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   ResultType result = TNL::abs( v1[ 0 ] - v2[ 0 ] );
   const Index n = v1.getSize();
#if defined( HAVE_OPENMP ) && _OPENMP >= 201107  // OpenMP 3.1 added support for min/max reduction operations
#pragma omp parallel for reduction(min:result) if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 1; i < n; i ++ )
      result =  min( result, (ResultType) TNL::abs( v1[ i ] - v2[ i ] ) );
   return result;
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorDifferenceL1Norm( const Vector1& v1,
                           const Vector2& v2 )
{
   typedef typename Vector1::IndexType Index;

   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   ResultType result( 0.0 );
   const Index n = v1.getSize();
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:result) if( TNL::Devices::Host::isOMPEnabled() &&n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i ++ )
      result += TNL::abs( v1[ i ] - v2[ i ] );
   return result;
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorDifferenceL2Norm( const Vector1& v1,
                           const Vector2& v2 )
{
   typedef typename Vector1::IndexType Index;

   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   ResultType result( 0.0 );
   const Index n = v1.getSize();
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:result) if( TNL::Devices::Host::isOMPEnabled() &&n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i ++ )
   {
      ResultType aux = TNL::abs( v1[ i ] - v2[ i ] );
      result += aux * aux;
   }
   return std::sqrt( result );
}


template< typename Vector1, typename Vector2, typename ResultType, typename Real_ >
ResultType
VectorOperations< Devices::Host >::
getVectorDifferenceLpNorm( const Vector1& v1,
                           const Vector2& v2,
                           const Real_ p )
{
   typedef typename Vector1::IndexType Index;

   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );
   TNL_ASSERT_GE( p, 1.0, "Parameter of the L^p norm must be at least 1.0." );

   if( p == 1.0 )
      return getVectorDifferenceL1Norm< Vector1, Vector2, ResultType >( v1, v2 );
   if( p == 2.0 )
      return getVectorDifferenceL2Norm< Vector1, Vector2, ResultType >( v1, v2 );

   ResultType result( 0.0 );
   const Index n = v1.getSize();
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:result) if( TNL::Devices::Host::isOMPEnabled() &&n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i ++ )
      result += std::pow( TNL::abs( v1.getElement( i ) - v2.getElement( i ) ), p );
   return std::pow( result, 1.0 / p );
}

template< typename Vector1, typename Vector2, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorDifferenceSum( const Vector1& v1,
                        const Vector2& v2 )
{
   typedef typename Vector1::IndexType Index;

   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   ResultType result( 0.0 );
   const Index n = v1.getSize();
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:result) if( TNL::Devices::Host::isOMPEnabled() &&n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i ++ )
      result += v1.getElement( i ) - v2.getElement( i );
   return result;
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
   typedef typename Vector1::IndexType Index;

   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );
   const Index n = v1.getSize();

#ifdef OPTIMIZED_VECTOR_HOST_OPERATIONS
#ifdef __GNUC__
   typedef typename Vector1::RealType Real;
   // We need to get the address of the first element to avoid
   // bounds checking in TNL::Array::operator[]
   const Real* V1 = v1.getData();
   const Real* V2 = v2.getData();
#endif

   ResultType dot1 = 0.0, dot2 = 0.0, dot3 = 0.0, dot4 = 0.0;
   Index i = 0;
   const Index unroll_limit = n - n % 4;
#ifdef HAVE_OPENMP
   #pragma omp parallel for \
      if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) \
      reduction(+:dot1,dot2,dot3,dot4) \
      lastprivate(i)
#endif
   for( i = 0; i < unroll_limit; i += 4 )
   {
#ifdef __GNUC__
      __builtin_prefetch(V1 + i + PrefetchDistance, 0, 0);
      __builtin_prefetch(V2 + i + PrefetchDistance, 0, 0);
#endif
      dot1 += v1[ i ]     * v2[ i ];
      dot2 += v1[ i + 1 ] * v2[ i + 1 ];
      dot3 += v1[ i + 2 ] * v2[ i + 2 ];
      dot4 += v1[ i + 3 ] * v2[ i + 3 ];
   }

   while( i < n )
   {
      dot1 += v1[ i ] * v2[ i ];
      i++;
   }

   return dot1 + dot2 + dot3 + dot4;

#else // OPTIMIZED_VECTOR_HOST_OPERATIONS

   ResultType result( 0.0 );
#ifdef HAVE_OPENMP
   #pragma omp parallel for reduction(+:result) if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i++ )
      result += v1[ i ] * v2[ i ];
   return result;
#endif // OPTIMIZED_VECTOR_HOST_OPERATIONS
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

#ifdef OPTIMIZED_VECTOR_HOST_OPERATIONS
#ifdef __GNUC__
   // We need to get the address of the first element to avoid
   // bounds checking in TNL::Array::operator[]
   typedef typename Vector1::RealType Real;   
         Real* Y = y.getData();
   const Real* X = x.getData();
#endif

   Index i = 0;
   const Index unroll_limit = n - n % 4;
#ifdef HAVE_OPENMP
   #pragma omp parallel for \
      if( n > OpenMPVectorOperationsThreshold ) \
      lastprivate(i)
#endif
   for(i = 0; i < unroll_limit; i += 4)
   {
#ifdef __GNUC__
      __builtin_prefetch(&y[ i + PrefetchDistance ], 1, 0);
      __builtin_prefetch(&x[ i + PrefetchDistance ], 0, 0);
#endif
      y[ i ]     = thisMultiplicator * y[ i ]     + alpha * x[ i ];
      y[ i + 1 ] = thisMultiplicator * y[ i + 1 ] + alpha * x[ i + 1 ];
      y[ i + 2 ] = thisMultiplicator * y[ i + 2 ] + alpha * x[ i + 2 ];
      y[ i + 3 ] = thisMultiplicator * y[ i + 3 ] + alpha * x[ i + 3 ];
   }

   while( i < n )
   {
      y[i] = thisMultiplicator * y[ i ] + alpha * x[ i ];
      i++;
   }

#else // OPTIMIZED_VECTOR_HOST_OPERATIONS

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
#endif // OPTIMIZED_VECTOR_HOST_OPERATIONS
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
