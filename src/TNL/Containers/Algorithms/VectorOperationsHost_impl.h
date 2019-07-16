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

static constexpr int OpenMPVectorOperationsThreshold = 512; // TODO: check this threshold
static constexpr int PrefetchDistance = 128;

template< typename Vector >
void
VectorOperations< Devices::Host >::
addElement( Vector& v,
            const typename Vector::IndexType i,
            const typename Vector::RealType& value )
{
   v[ i ] += value;
}

template< typename Vector, typename Scalar >
void
VectorOperations< Devices::Host >::
addElement( Vector& v,
            const typename Vector::IndexType i,
            const typename Vector::RealType& value,
            const Scalar thisElementMultiplicator )
{
   v[ i ] = thisElementMultiplicator * v[ i ] + value;
}



template< typename Vector, typename Scalar >
void
VectorOperations< Devices::Host >::
vectorScalarMultiplication( Vector& v,
                            const Scalar alpha )
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


template< typename Vector1, typename Vector2, typename Scalar1, typename Scalar2 >
void
VectorOperations< Devices::Host >::
addVector( Vector1& y,
           const Vector2& x,
           const Scalar1 alpha,
           const Scalar2 thisMultiplicator )
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

template< typename Vector1, typename Vector2, typename Vector3,
          typename Scalar1, typename Scalar2, typename Scalar3 >
void
VectorOperations< Devices::Host >::
addVectors( Vector1& v,
            const Vector2& v1,
            const Scalar1 multiplicator1,
            const Vector3& v2,
            const Scalar2 multiplicator2,
            const Scalar3 thisMultiplicator )
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

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Host >::
getVectorSum( const Vector& v )
{
   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   if( std::is_same< ResultType, bool >::value )
      abort();

   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

   const auto* data = v.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i )  -> ResultType { return  data[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a += b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a += b; };
   return Reduction< Devices::Host >::reduce( v.getSize(), reduction, volatileReduction, fetch, ( ResultType ) 0 );
}

template< Algorithms::PrefixSumType Type, typename Vector >
void
VectorOperations< Devices::Host >::
prefixSum( Vector& v,
           typename Vector::IndexType begin,
           typename Vector::IndexType end )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

   auto reduction = [=] __cuda_callable__ ( RealType& a, const RealType& b ) { a += b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile RealType& a, volatile RealType& b ) { a += b; };

   PrefixSum< Devices::Host, Type >::perform( v, begin, end, reduction, volatileReduction, ( RealType ) 0.0 );
}

template< Algorithms::PrefixSumType Type, typename Vector, typename Flags >
void
VectorOperations< Devices::Host >::
segmentedPrefixSum( Vector& v,
                    Flags& f,
                    typename Vector::IndexType begin,
                    typename Vector::IndexType end )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

   auto reduction = [=] __cuda_callable__ ( RealType& a, const RealType& b ) { a += b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile RealType& a, volatile RealType& b ) { a += b; };

   SegmentedPrefixSum< Devices::Host, Type >::perform( v, f, begin, end, reduction, volatileReduction, ( RealType ) 0.0 );
}

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
