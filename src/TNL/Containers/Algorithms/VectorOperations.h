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
