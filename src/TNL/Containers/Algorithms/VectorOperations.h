/***************************************************************************
                          VectorOperations.h  -  description
                             -------------------
    begin                : Nov 8, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Algorithms/PrefixSum.h>
#include <TNL/Containers/Algorithms/Reduction.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename Device >
class VectorOperations{};

template<>
class VectorOperations< Devices::Host >
{
public:
   template< typename Vector >
   static void addElement( Vector& v,
                           const typename Vector::IndexType i,
                           const typename Vector::RealType& value );

   template< typename Vector, typename Scalar >
   static void addElement( Vector& v,
                           const typename Vector::IndexType i,
                           const typename Vector::RealType& value,
                           const Scalar thisElementMultiplicator );

   template< typename Vector, typename Scalar >
   static void vectorScalarMultiplication( Vector& v, Scalar alpha );

   template< typename Vector1, typename Vector2, typename Scalar1, typename Scalar2 >
   static void addVector( Vector1& y,
                          const Vector2& x,
                          const Scalar1 alpha,
                          const Scalar2 thisMultiplicator = 1.0 );

   template< typename Vector1, typename Vector2, typename Vector3, typename Scalar1, typename Scalar2, typename Scalar3 >
   static void addVectors( Vector1& v,
                           const Vector2& v1,
                           const Scalar1 multiplicator1,
                           const Vector3& v2,
                           const Scalar2 multiplicator2,
                           const Scalar3 thisMultiplicator = 1.0 );

   template< typename Vector, typename ResultType = typename Vector::RealType >
   static ResultType getVectorSum( const Vector& v );

   template< Algorithms::PrefixSumType Type,
             typename Vector >
   static void prefixSum( Vector& v,
                          const typename Vector::IndexType begin,
                          const typename Vector::IndexType end );

   template< Algorithms::PrefixSumType Type, typename Vector, typename Flags >
   static void segmentedPrefixSum( Vector& v,
                                   Flags& f,
                                   const typename Vector::IndexType begin,
                                   const typename Vector::IndexType end );
};

template<>
class VectorOperations< Devices::Cuda >
{
public:
   template< typename Vector >
   static void addElement( Vector& v,
                           const typename Vector::IndexType i,
                           const typename Vector::RealType& value );

   template< typename Vector, typename Scalar >
   static void addElement( Vector& v,
                           const typename Vector::IndexType i,
                           const typename Vector::RealType& value,
                           const Scalar thisElementMultiplicator );

   template< typename Vector, typename Scalar >
   static void vectorScalarMultiplication( Vector& v, const Scalar alpha );

   template< typename Vector1, typename Vector2, typename Scalar1, typename Scalar2 >
   static void addVector( Vector1& y,
                          const Vector2& x,
                          const Scalar1 alpha,
                          const Scalar2 thisMultiplicator = 1.0 );

   template< typename Vector1, typename Vector2, typename Vector3, typename Scalar1, typename Scalar2, typename Scalar3 >
   static void addVectors( Vector1& v,
                           const Vector2& v1,
                           const Scalar1 multiplicator1,
                           const Vector3& v2,
                           const Scalar2 multiplicator2,
                           const Scalar3 thisMultiplicator = 1.0 );

   template< typename Vector, typename ResultType = typename Vector::RealType >
   static ResultType getVectorSum( const Vector& v );

   template< Algorithms::PrefixSumType Type,
             typename Vector >
   static void prefixSum( Vector& v,
                          const typename Vector::IndexType begin,
                          const typename Vector::IndexType end );

   template< Algorithms::PrefixSumType Type, typename Vector, typename Flags >
   static void segmentedPrefixSum( Vector& v,
                                   Flags& f,
                                   const typename Vector::IndexType begin,
                                   const typename Vector::IndexType end );
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL

#include <TNL/Containers/Algorithms/VectorOperationsHost_impl.h>
#include <TNL/Containers/Algorithms/VectorOperationsCuda_impl.h>
