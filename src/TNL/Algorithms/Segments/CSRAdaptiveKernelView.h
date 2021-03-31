/***************************************************************************
                          CSRAdaptiveKernelView.h -  description
                             -------------------
    begin                : Feb 7, 2021 -> Joe Biden inauguration
    copyright            : (C) 2021 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/detail/CSRAdaptiveKernelBlockDescriptor.h>
#include <TNL/Algorithms/Segments/detail/CSRAdaptiveKernelParameters.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {

template< typename Index,
          typename Device >
struct CSRAdaptiveKernelView
{
   using IndexType = Index;
   using DeviceType = Device;
   using ViewType = CSRAdaptiveKernelView< Index, Device >;
   using ConstViewType = CSRAdaptiveKernelView< Index, Device >;
   using BlocksType = TNL::Containers::Vector< detail::CSRAdaptiveKernelBlockDescriptor< Index >, Device, Index >;
   using BlocksView = typename BlocksType::ViewType;

   static constexpr int MaxValueSizeLog = detail::CSRAdaptiveKernelParameters<>::MaxValueSizeLog;

   static int getSizeValueLog( const int& i ) { return detail::CSRAdaptiveKernelParameters<>::getSizeValueLog( i ); };

   CSRAdaptiveKernelView() = default;

   void setBlocks( BlocksType& blocks, const int idx );

   ViewType getView();

   ConstViewType getConstView() const;

   static TNL::String getKernelType();

   template< typename OffsetsView,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Real,
             typename... Args >
   void reduceSegments( const OffsetsView& offsets,
                        Index first,
                        Index last,
                        Fetch& fetch,
                        const Reduction& reduction,
                        ResultKeeper& keeper,
                        const Real& zero,
                        Args... args ) const;

   CSRAdaptiveKernelView& operator=( const CSRAdaptiveKernelView< Index, Device >& kernelView );

   void printBlocks( int idx ) const;

   protected:
      BlocksView blocksArray[ MaxValueSizeLog ];
};

      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL

#include <TNL/Algorithms/Segments/CSRAdaptiveKernelView.hpp>
