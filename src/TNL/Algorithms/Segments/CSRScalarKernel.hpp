/***************************************************************************
                          CSRScalarKernel.h -  description
                             -------------------
    begin                : Jan 23, 2021 -> Joe Biden inauguration
    copyright            : (C) 2021 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>
#include <TNL/Cuda/LaunchHelpers.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Algorithms/Segments/CSRScalarKernel.h>
#include <TNL/Algorithms/Segments/details/LambdaAdapter.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {

template< typename Index,
          typename Device >
    template< typename Offsets >
void
CSRScalarKernel< Index, Device >::
init( const Offsets& offsets )
{
}

template< typename Index,
          typename Device >
void
CSRScalarKernel< Index, Device >::
reset()
{
}

template< typename Index,
          typename Device >
auto
CSRScalarKernel< Index, Device >::
getView() -> ViewType
{
    return *this;
}

template< typename Index,
          typename Device >
auto
CSRScalarKernel< Index, Device >::
getConstView() const -> ConstViewType
{
    return *this;
};

template< typename Index,
          typename Device >
TNL::String
CSRScalarKernel< Index, Device >::
getKernelType()
{
    return "Scalar";
}

template< typename Index,
          typename Device >
    template< typename OffsetsView,
              typename Fetch,
              typename Reduction,
              typename ResultKeeper,
              typename Real,
              typename... Args >
void
CSRScalarKernel< Index, Device >::
segmentsReduction( const OffsetsView& offsets,
                   Index first,
                   Index last,
                   Fetch& fetch,
                   const Reduction& reduction,
                   ResultKeeper& keeper,
                   const Real& zero,
                   Args... args )
{
    auto l = [=] __cuda_callable__ ( const IndexType segmentIdx, Args... args ) mutable {
        const IndexType begin = offsets[ segmentIdx ];
        const IndexType end = offsets[ segmentIdx + 1 ];
        Real aux( zero );
        IndexType localIdx( 0 );
        bool compute( true );
        for( IndexType globalIdx = begin; globalIdx < end && compute; globalIdx++  )
            aux = reduction( aux, details::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx, compute ) );
        keeper( segmentIdx, aux );
    };
    Algorithms::ParallelFor< Device >::exec( first, last, l, args... );
}
      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL
