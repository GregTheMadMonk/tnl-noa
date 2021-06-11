/***************************************************************************
                          CSRHybridKernel.h -  description
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
#include <TNL/Algorithms/Segments/detail/LambdaAdapter.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {

template< typename Index,
          typename Device,
          int ThreadsInBlock = 128 >
struct CSRHybridKernel
{
   using IndexType = Index;
   using DeviceType = Device;
   using ViewType = CSRHybridKernel< Index, Device, ThreadsInBlock >;
   using ConstViewType = CSRHybridKernel< Index, Device, ThreadsInBlock >;

   template< typename Offsets >
   void init( const Offsets& offsets );

   void reset();

   ViewType getView();

   ConstViewType getConstView() const;

   static TNL::String getKernelType();

   template< typename OffsetsView,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Real >
   void reduceSegments( const OffsetsView& offsets,
                                  Index first,
                                  Index last,
                                  Fetch& fetch,
                                  const Reduction& reduction,
                                  ResultKeeper& keeper,
                                  const Real& zero ) const;

   protected:
      int threadsPerSegment;
};

      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL

#include <TNL/Algorithms/Segments/Kernels/CSRHybridKernel.hpp>
