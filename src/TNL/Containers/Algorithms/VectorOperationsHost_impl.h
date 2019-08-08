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
