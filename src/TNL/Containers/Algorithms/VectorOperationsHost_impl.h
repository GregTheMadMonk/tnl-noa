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


/*template< typename Vector1, typename Vector2, typename ResultType >
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
}*/

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
