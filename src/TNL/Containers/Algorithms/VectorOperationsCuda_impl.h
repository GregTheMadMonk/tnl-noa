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
#include <TNL/Containers/Algorithms/CudaPrefixSumKernel.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Cuda >::
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
   return Reduction< Devices::Cuda >::reduce( v.getSize(), reduction, volatileReduction, fetch, ( ResultType ) 0 );
}

template< Algorithms::PrefixSumType Type,
          typename Vector >
void
VectorOperations< Devices::Cuda >::
prefixSum( Vector& v,
           typename Vector::IndexType begin,
           typename Vector::IndexType end )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

   auto reduction = [=] __cuda_callable__ ( RealType& a, const RealType& b ) { a += b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile RealType& a, volatile RealType& b ) { a += b; };

   PrefixSum< Devices::Cuda, Type >::perform( v, begin, end, reduction, volatileReduction, ( RealType ) 0.0 );
}

template< Algorithms::PrefixSumType Type, typename Vector, typename Flags >
void
VectorOperations< Devices::Cuda >::
segmentedPrefixSum( Vector& v,
                    Flags& f,
                    typename Vector::IndexType begin,
                    typename Vector::IndexType end )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

   auto reduction = [=] __cuda_callable__ ( RealType& a, const RealType& b ) { a += b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile RealType& a, volatile RealType& b ) { a += b; };

   SegmentedPrefixSum< Devices::Cuda, Type >::perform( v, f, begin, end, reduction, volatileReduction, ( RealType ) 0.0 );
}


} // namespace Algorithms
} // namespace Containers
} // namespace TNL
