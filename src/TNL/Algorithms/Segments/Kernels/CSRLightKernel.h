/***************************************************************************
                          CSRLightKernel.h -  description
                             -------------------
    begin                : Jun 9, 2021 -> Joe Biden inauguration
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

enum LightCSRSThreadsMapping { LightCSRConstantThreads, CSRLightAutomaticThreads, CSRLightAutomaticThreadsLightSpMV };

template< typename Index,
          typename Device >
struct CSRLightKernel
{
   using IndexType = Index;
   using DeviceType = Device;
   using ViewType = CSRLightKernel< Index, Device >;
   using ConstViewType = CSRLightKernel< Index, Device >;

   template< typename Offsets >
   void init( const Offsets& offsets );

   void reset();

   ViewType getView();

   ConstViewType getConstView() const;

   static TNL::String getKernelType();

   TNL::String getSetup() const;

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


   void setThreadsMapping( LightCSRSThreadsMapping mapping );

   LightCSRSThreadsMapping getThreadsMapping() const;

   void setThreadsPerSegment( int threadsPerSegment );

   int getThreadsPerSegment() const;

   protected:

      LightCSRSThreadsMapping mapping = CSRLightAutomaticThreads;

      int threadsPerSegment = 32;
};

      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL

#include <TNL/Algorithms/Segments/Kernels/CSRLightKernel.hpp>
