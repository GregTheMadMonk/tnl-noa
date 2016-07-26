/***************************************************************************
                          VectorOperationsHost_impl.h  -  description
                             -------------------
    begin                : Nov 8, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

namespace TNL {
namespace Vectors {   

static const int OpenMPVectorOperationsThreshold = 65536; // TODO: check this threshold

template< typename Vector >
void VectorOperations< tnlHost >::addElement( Vector& v,
                                                 const typename Vector::IndexType i,
                                                 const typename Vector::RealType& value )
{
   v[ i ] += value;
}

template< typename Vector >
void VectorOperations< tnlHost >::addElement( Vector& v,
                                                 const typename Vector::IndexType i,
                                                 const typename Vector::RealType& value,
                                                 const typename Vector::RealType& thisElementMultiplicator )
{
   v[ i ] = thisElementMultiplicator * v[ i ] + value;
}

template< typename Vector >
typename Vector::RealType VectorOperations< tnlHost >::getVectorMax( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;
   Assert( v. getSize() > 0, );
   Real result = v. getElement( 0 );
   const Index n = v. getSize();
   for( Index i = 1; i < n; i ++ )
      result = max( result, v. getElement( i ) );
   return result;
}

template< typename Vector >
typename Vector :: RealType VectorOperations< tnlHost > :: getVectorMin( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;
   Assert( v. getSize() > 0, );
   Real result = v. getElement( 0 );
   const Index n = v. getSize();
   for( Index i = 1; i < n; i ++ )
      result = min( result, v. getElement( i ) );
   return result;
}

template< typename Vector >
typename Vector :: RealType VectorOperations< tnlHost > :: getVectorAbsMax( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;
   Assert( v. getSize() > 0, );
   Real result = std::fabs( v. getElement( 0 ) );
   const Index n = v. getSize();
   for( Index i = 1; i < n; i ++ )
      result = max( result, ( Real ) std::fabs( v. getElement( i ) ) );
   return result;
}


template< typename Vector >
typename Vector :: RealType VectorOperations< tnlHost > :: getVectorAbsMin( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;
   Assert( v. getSize() > 0, );
   Real result = std::fabs( v. getElement( 0 ) );
   const Index n = v. getSize();
   for( Index i = 1; i < n; i ++ )
      result = min( result, ( Real ) std::fabs( v. getElement( i ) ) );
   return result;
}

template< typename Vector >
typename Vector::RealType
VectorOperations< tnlHost >::
getVectorL1Norm( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;
   Assert( v. getSize() > 0, );

   Real result( 0.0 );
   const Index n = v. getSize();
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:result) if( TNL::tnlHost::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i ++ )
      result += std::fabs( v[ i ] );
   return result;
}

template< typename Vector >
typename Vector::RealType
VectorOperations< tnlHost >::
getVectorL2Norm( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;
   Assert( v. getSize() > 0, );
   Real result( 0.0 );
   const Index n = v. getSize();
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:result) if( TNL::tnlHost::isOMPEnabled() &&n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i ++ )
   {
      const Real& aux = v[ i ];
      result += aux * aux;
   }
   return std::sqrt( result );
}

template< typename Vector >
typename Vector::RealType
VectorOperations< tnlHost >::
getVectorLpNorm( const Vector& v,
                 const typename Vector :: RealType& p )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;
   Assert( v. getSize() > 0, );
   Assert( p > 0.0,
              std::cerr << " p = " << p );
   if( p == 1.0 )
      return getVectorL1Norm( v );
   if( p == 2.0 )
      return getVectorL2Norm( v );

   Real result( 0.0 );
   const Index n = v. getSize();
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:result) if( TNL::tnlHost::isOMPEnabled() &&n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i ++ )
      result += std::pow( std::fabs( v[ i ] ), p );
   return std::pow( result, 1.0 / p );
}

template< typename Vector >
typename Vector :: RealType VectorOperations< tnlHost > :: getVectorSum( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;
   Assert( v. getSize() > 0, );

   Real result( 0.0 );
   const Index n = v. getSize();
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:result) if( TNL::tnlHost::isOMPEnabled() &&n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i ++ )
      result += v[ i ];
   return result;
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType VectorOperations< tnlHost > :: getVectorDifferenceMax( const Vector1& v1,
                                                                                       const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;
   Assert( v1. getSize() > 0, );
   Assert( v1. getSize() == v2. getSize(), );
   Real result = v1. getElement( 0 ) - v2. getElement( 0 );
   const Index n = v1. getSize();
   for( Index i = 1; i < n; i ++ )
      result =  max( result, v1. getElement( i ) - v2. getElement( i ) );
   return result;
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType VectorOperations< tnlHost > :: getVectorDifferenceMin( const Vector1& v1,
                                                                                       const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( v1. getSize() > 0, );
   Assert( v1. getSize() == v2. getSize(), );

   Real result = v1. getElement( 0 ) - v2. getElement( 0 );
   const Index n = v1. getSize();
   for( Index i = 1; i < n; i ++ )
      result =  min( result, v1. getElement( i ) - v2. getElement( i ) );
   return result;
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType VectorOperations< tnlHost > :: getVectorDifferenceAbsMax( const Vector1& v1,
                                                                                          const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( v1. getSize() > 0, );
   Assert( v1. getSize() == v2. getSize(), );

   Real result = std::fabs( v1. getElement( 0 ) - v2. getElement( 0 ) );
   const Index n = v1. getSize();
   for( Index i = 1; i < n; i ++ )
      result =  max( result, ( Real ) std::fabs( v1. getElement( i ) - v2. getElement( i ) ) );
   return result;
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType VectorOperations< tnlHost > :: getVectorDifferenceAbsMin( const Vector1& v1,
                                                                                          const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( v1. getSize() > 0, );
   Assert( v1. getSize() == v2. getSize(), );

   Real result = std::fabs( v1[ 0 ] - v2[ 0 ] );
   const Index n = v1. getSize();
   for( Index i = 1; i < n; i ++ )
      result =  min( result, ( Real ) std::fabs( v1[ i ] - v2[ i ] ) );
   return result;
}

template< typename Vector1, typename Vector2 >
typename Vector1::RealType
VectorOperations< tnlHost >::
getVectorDifferenceL1Norm( const Vector1& v1,
                           const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( v1. getSize() > 0, );
   Assert( v1. getSize() == v2. getSize(), );

   Real result( 0.0 );
   const Index n = v1. getSize();
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:result) if( TNL::tnlHost::isOMPEnabled() &&n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i ++ )
      result += std::fabs( v1[ i ] - v2[ i ] );
   return result;
}

template< typename Vector1, typename Vector2 >
typename Vector1::RealType
VectorOperations< tnlHost >::
getVectorDifferenceL2Norm( const Vector1& v1,
                           const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( v1. getSize() > 0, );
   Assert( v1. getSize() == v2. getSize(), );

   Real result( 0.0 );
   const Index n = v1. getSize();
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:result) if( TNL::tnlHost::isOMPEnabled() &&n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i ++ )
   {
      Real aux = std::fabs( v1[ i ] - v2[ i ] );
      result += aux * aux;
   }
   return std::sqrt( result );
}


template< typename Vector1, typename Vector2 >
typename Vector1::RealType
VectorOperations< tnlHost >::
getVectorDifferenceLpNorm( const Vector1& v1,
                           const Vector2& v2,
                           const typename Vector1::RealType& p )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( p > 0.0,
              std::cerr << " p = " << p );
   Assert( v1. getSize() > 0, );
   Assert( v1. getSize() == v2. getSize(), );

   if( p == 1.0 )
      return getVectorDifferenceL1Norm( v1, v2 );
   if( p == 2.0 )
      return getVectorDifferenceL2Norm( v1, v2 );

   Real result( 0.0 );
   const Index n = v1. getSize();
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:result) if( TNL::tnlHost::isOMPEnabled() &&n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i ++ )
      result += std::pow( std::fabs( v1. getElement( i ) - v2. getElement( i ) ), p );
   return std::pow( result, 1.0 / p );
}

template< typename Vector1, typename Vector2 >
typename Vector1::RealType VectorOperations< tnlHost > :: getVectorDifferenceSum( const Vector1& v1,
                                                                                     const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( v1. getSize() > 0, );
   Assert( v1. getSize() == v2. getSize(), );

   Real result( 0.0 );
   const Index n = v1. getSize();
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:result) if( TNL::tnlHost::isOMPEnabled() &&n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i ++ )
      result += v1. getElement( i ) - v2. getElement( i );
   return result;
}


template< typename Vector >
void VectorOperations< tnlHost > :: vectorScalarMultiplication( Vector& v,
                                                                   const typename Vector :: RealType& alpha )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;

   Assert( v. getSize() > 0, );

   const Index n = v. getSize();
#ifdef HAVE_OPENMP
#pragma omp parallel for if( n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i ++ )
      v[ i ] *= alpha;
}


template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType VectorOperations< tnlHost > :: getScalarProduct( const Vector1& v1,
                                                                                 const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( v1. getSize() > 0, );
   Assert( v1. getSize() == v2. getSize(), );

   Real result( 0.0 );
   const Index n = v1. getSize();
#ifdef HAVE_OPENMP
  #pragma omp parallel for reduction(+:result) if( TNL::tnlHost::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
   for( Index i = 0; i < n; i++ )
      result += v1[ i ] * v2[ i ];
   /*Real result1( 0.0 ), result2( 0.0 ), result3( 0.0 ), result4( 0.0 ),
        result5( 0.0 ), result6( 0.0 ), result7( 0.0 ), result8( 0.0 );
   Index i( 0 );
   while( i + 8 < n )
   {
      result1 += v1[ i ] * v2[ i ];
      result2 += v1[ i + 1 ] * v2[ i + 1 ];
      result3 += v1[ i + 2 ] * v2[ i + 2 ];
      result4 += v1[ i + 3 ] * v2[ i + 3 ];
      result5 += v1[ i + 4 ] * v2[ i + 4 ];
      result6 += v1[ i + 5 ] * v2[ i + 5 ];
      result7 += v1[ i + 6 ] * v2[ i + 6 ];
      result8 += v1[ i + 7 ] * v2[ i + 7 ];
      i += 8;
   }
   Real result = result1 + result2 + result3 + result4 + result5 +result6 +result7 +result8;
   while( i < n )
      result += v1[ i ] * v2[ i++ ];*/
   return result;
}

template< typename Vector1, typename Vector2 >
void VectorOperations< tnlHost > :: addVector( Vector1& y,
                                                  const Vector2& x,
                                                  const typename Vector2::RealType& alpha,
                                                  const typename Vector1::RealType& thisMultiplicator )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   Assert( x. getSize() > 0, );
   Assert( x. getSize() == y. getSize(), );

   const Index n = y. getSize();
   if( thisMultiplicator == 1.0 )
#ifdef HAVE_OPENMP
#pragma omp parallel for if( TNL::tnlHost::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
      for( Index i = 0; i < n; i ++ )
         y[ i ] += alpha * x[ i ];
   else
#ifdef HAVE_OPENMP
#pragma omp parallel for if( TNL::tnlHost::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
      for( Index i = 0; i < n; i ++ )
         y[ i ] = thisMultiplicator * y[ i ] + alpha * x[ i ];
}

template< typename Vector1,
          typename Vector2,
          typename Vector3 >
void
VectorOperations< tnlHost >::
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
 
   const Index n = v.getSize();
   if( thisMultiplicator == 1.0 )
#ifdef HAVE_OPENMP
#pragma omp parallel for if( TNL::tnlHost::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
      for( Index i = 0; i < n; i ++ )
         v[ i ] += multiplicator1 * v1[ i ] + multiplicator2 * v2[ i ];
   else
#ifdef HAVE_OPENMP
#pragma omp parallel for if( TNL::tnlHost::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif
      for( Index i = 0; i < n; i ++ )
         v[ i ] = thisMultiplicator * v[ i ] + multiplicator1 * v1[ i ] + multiplicator2 * v2[ i ];
}

template< typename Vector >
void VectorOperations< tnlHost >::computePrefixSum( Vector& v,
                                                       typename Vector::IndexType begin,
                                                       typename Vector::IndexType end )
{
   typedef typename Vector::IndexType Index;
   for( Index i = begin + 1; i < end; i++ )
      v[ i ] += v[ i - 1 ];
}

template< typename Vector >
void VectorOperations< tnlHost >::computeExclusivePrefixSum( Vector& v,
                                                                typename Vector::IndexType begin,
                                                                typename Vector::IndexType end )
{
   typedef typename Vector::IndexType Index;
   typedef typename Vector::RealType Real;
   Real aux( v[ begin ] );
   v[ begin ] = 0.0;
   for( Index i = begin + 1; i < end; i++ )
   {
      Real x = v[ i ];
      v[ i ] = aux;
      aux += x;
   }
}

} // namespace Vectors
} //namespace TNL

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#include <TNL/Vectors/Vector.h>

namespace TNL {
namespace Vectors {   

/****
 * Max
 */
extern template int         VectorOperations< tnlHost >::getVectorMax( const Vector< int, tnlHost, int >& v );
extern template long int    VectorOperations< tnlHost >::getVectorMax( const Vector< long int, tnlHost, int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorMax( const Vector< float, tnlHost, int >& v );
#endif
extern template double      VectorOperations< tnlHost >::getVectorMax( const Vector< double, tnlHost, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorMax( const Vector< long double, tnlHost, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template int         VectorOperations< tnlHost >::getVectorMax( const Vector< int, tnlHost, long int >& v );
extern template long int    VectorOperations< tnlHost >::getVectorMax( const Vector< long int, tnlHost, long int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorMax( const Vector< float, tnlHost, long int >& v );
#endif
extern template double      VectorOperations< tnlHost >::getVectorMax( const Vector< double, tnlHost, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorMax( const Vector< long double, tnlHost, long int >& v );
#endif
#endif

/****
 * Min
 */
extern template int         VectorOperations< tnlHost >::getVectorMin( const Vector< int, tnlHost, int >& v );
extern template long int    VectorOperations< tnlHost >::getVectorMin( const Vector< long int, tnlHost, int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorMin( const Vector< float, tnlHost, int >& v );
#endif
extern template double      VectorOperations< tnlHost >::getVectorMin( const Vector< double, tnlHost, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorMin( const Vector< long double, tnlHost, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template int         VectorOperations< tnlHost >::getVectorMin( const Vector< int, tnlHost, long int >& v );
extern template long int    VectorOperations< tnlHost >::getVectorMin( const Vector< long int, tnlHost, long int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorMin( const Vector< float, tnlHost, long int >& v );
#endif
extern template double      VectorOperations< tnlHost >::getVectorMin( const Vector< double, tnlHost, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorMin( const Vector< long double, tnlHost, long int >& v );
#endif
#endif

/****
 * Abs max
 */
extern template int         VectorOperations< tnlHost >::getVectorAbsMax( const Vector< int, tnlHost, int >& v );
extern template long int    VectorOperations< tnlHost >::getVectorAbsMax( const Vector< long int, tnlHost, int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorAbsMax( const Vector< float, tnlHost, int >& v );
#endif
extern template double      VectorOperations< tnlHost >::getVectorAbsMax( const Vector< double, tnlHost, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorAbsMax( const Vector< long double, tnlHost, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template int         VectorOperations< tnlHost >::getVectorAbsMax( const Vector< int, tnlHost, long int >& v );
extern template long int    VectorOperations< tnlHost >::getVectorAbsMax( const Vector< long int, tnlHost, long int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorAbsMax( const Vector< float, tnlHost, long int >& v );
#endif
extern template double      VectorOperations< tnlHost >::getVectorAbsMax( const Vector< double, tnlHost, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorAbsMax( const Vector< long double, tnlHost, long int >& v );
#endif
#endif

/****
 * Abs min
 */
extern template int         VectorOperations< tnlHost >::getVectorAbsMin( const Vector< int, tnlHost, int >& v );
extern template long int    VectorOperations< tnlHost >::getVectorAbsMin( const Vector< long int, tnlHost, int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorAbsMin( const Vector< float, tnlHost, int >& v );
#endif
extern template double      VectorOperations< tnlHost >::getVectorAbsMin( const Vector< double, tnlHost, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorAbsMin( const Vector< long double, tnlHost, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template int         VectorOperations< tnlHost >::getVectorAbsMin( const Vector< int, tnlHost, long int >& v );
extern template long int    VectorOperations< tnlHost >::getVectorAbsMin( const Vector< long int, tnlHost, long int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorAbsMin( const Vector< float, tnlHost, long int >& v );
#endif
extern template double      VectorOperations< tnlHost >::getVectorAbsMin( const Vector< double, tnlHost, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorAbsMin( const Vector< long double, tnlHost, long int >& v );
#endif
#endif

/****
 * Lp norm
 */
extern template int         VectorOperations< tnlHost >::getVectorLpNorm( const Vector< int, tnlHost, int >& v, const int& p );
extern template long int    VectorOperations< tnlHost >::getVectorLpNorm( const Vector< long int, tnlHost, int >& v, const long int& p );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorLpNorm( const Vector< float, tnlHost, int >& v, const float& p );
#endif
extern template double      VectorOperations< tnlHost >::getVectorLpNorm( const Vector< double, tnlHost, int >& v, const double& p );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorLpNorm( const Vector< long double, tnlHost, int >& v, const long double& p );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template int         VectorOperations< tnlHost >::getVectorLpNorm( const Vector< int, tnlHost, long int >& v, const int& p );
extern template long int    VectorOperations< tnlHost >::getVectorLpNorm( const Vector< long int, tnlHost, long int >& v, const long int& p );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorLpNorm( const Vector< float, tnlHost, long int >& v, const float& p );
#endif
extern template double      VectorOperations< tnlHost >::getVectorLpNorm( const Vector< double, tnlHost, long int >& v, const double& p );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorLpNorm( const Vector< long double, tnlHost, long int >& v, const long double& p );
#endif
#endif

/****
 * Sum
 */
extern template int         VectorOperations< tnlHost >::getVectorSum( const Vector< int, tnlHost, int >& v );
extern template long int    VectorOperations< tnlHost >::getVectorSum( const Vector< long int, tnlHost, int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorSum( const Vector< float, tnlHost, int >& v );
#endif
extern template double      VectorOperations< tnlHost >::getVectorSum( const Vector< double, tnlHost, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorSum( const Vector< long double, tnlHost, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template int         VectorOperations< tnlHost >::getVectorSum( const Vector< int, tnlHost, long int >& v );
extern template long int    VectorOperations< tnlHost >::getVectorSum( const Vector< long int, tnlHost, long int >& v );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorSum( const Vector< float, tnlHost, long int >& v );
#endif
extern template double      VectorOperations< tnlHost >::getVectorSum( const Vector< double, tnlHost, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorSum( const Vector< long double, tnlHost, long int >& v );
#endif
#endif

/****
 * Difference max
 */
extern template int         VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< int, tnlHost, int >& v1, const Vector< int, tnlHost, int >& v2 );
extern template long int    VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< long int, tnlHost, int >& v1, const Vector< long int, tnlHost, int >& v2 );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< float, tnlHost, int >& v1,  const Vector< float, tnlHost, int >& v2);
#endif
extern template double      VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< double, tnlHost, int >& v1, const Vector< double, tnlHost, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< long double, tnlHost, int >& v1, const Vector< long double, tnlHost, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template int         VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< int, tnlHost, long int >& v1, const Vector< int, tnlHost, long int >& v2 );
extern template long int    VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< long int, tnlHost, long int >& v1, const Vector< long int, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< float, tnlHost, long int >& v1, const Vector< float, tnlHost, long int >& v2 );
#endif
extern template double      VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< double, tnlHost, long int >& v1, const Vector< double, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< long double, tnlHost, long int >& v1, const Vector< long double, tnlHost, long int >& v2 );
#endif
#endif

/****
 * Difference min
 */
extern template int         VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< int, tnlHost, int >& v1, const Vector< int, tnlHost, int >& v2 );
extern template long int    VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< long int, tnlHost, int >& v1, const Vector< long int, tnlHost, int >& v2 );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< float, tnlHost, int >& v1,  const Vector< float, tnlHost, int >& v2);
#endif
extern template double      VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< double, tnlHost, int >& v1, const Vector< double, tnlHost, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< long double, tnlHost, int >& v1, const Vector< long double, tnlHost, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template int         VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< int, tnlHost, long int >& v1, const Vector< int, tnlHost, long int >& v2 );
extern template long int    VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< long int, tnlHost, long int >& v1, const Vector< long int, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< float, tnlHost, long int >& v1, const Vector< float, tnlHost, long int >& v2 );
#endif
extern template double      VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< double, tnlHost, long int >& v1, const Vector< double, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< long double, tnlHost, long int >& v1, const Vector< long double, tnlHost, long int >& v2 );
#endif
#endif

/****
 * Difference abs max
 */
extern template int         VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< int, tnlHost, int >& v1, const Vector< int, tnlHost, int >& v2 );
extern template long int    VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< long int, tnlHost, int >& v1, const Vector< long int, tnlHost, int >& v2 );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< float, tnlHost, int >& v1,  const Vector< float, tnlHost, int >& v2);
#endif
extern template double      VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< double, tnlHost, int >& v1, const Vector< double, tnlHost, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< long double, tnlHost, int >& v1, const Vector< long double, tnlHost, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template int         VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< int, tnlHost, long int >& v1, const Vector< int, tnlHost, long int >& v2 );
extern template long int    VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< long int, tnlHost, long int >& v1, const Vector< long int, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< float, tnlHost, long int >& v1, const Vector< float, tnlHost, long int >& v2 );
#endif
extern template double      VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< double, tnlHost, long int >& v1, const Vector< double, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< long double, tnlHost, long int >& v1, const Vector< long double, tnlHost, long int >& v2 );
#endif
#endif

/****
 * Difference abs min
 */
extern template int         VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< int, tnlHost, int >& v1, const Vector< int, tnlHost, int >& v2 );
extern template long int    VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< long int, tnlHost, int >& v1, const Vector< long int, tnlHost, int >& v2 );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< float, tnlHost, int >& v1,  const Vector< float, tnlHost, int >& v2);
#endif
extern template double      VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< double, tnlHost, int >& v1, const Vector< double, tnlHost, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< long double, tnlHost, int >& v1, const Vector< long double, tnlHost, int >& v2 );
#endif


#ifdef INSTANTIATE_LONG_INT
extern template int         VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< int, tnlHost, long int >& v1, const Vector< int, tnlHost, long int >& v2 );
extern template long int    VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< long int, tnlHost, long int >& v1, const Vector< long int, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
extern template float       VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< float, tnlHost, long int >& v1, const Vector< float, tnlHost, long int >& v2 );
#endif
extern template double      VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< double, tnlHost, long int >& v1, const Vector< double, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
extern template long double VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< long double, tnlHost, long int >& v1, const Vector< long double, tnlHost, long int >& v2 );
#endif
#endif

} // namespace Vectors
} // namespace TNL
#endif


