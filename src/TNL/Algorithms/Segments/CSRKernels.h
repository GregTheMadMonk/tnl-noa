/***************************************************************************
                          CSRKernels.h -  description
                             -------------------
    begin                : Jan 20, 2021 -> Joe Biden inauguration
    copyright            : (C) 2021 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>
#include <TNL/Cuda/LaunchHelpers.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Algorithms/Segments/details/LambdaAdapter.h>

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

    ViewType getView() { return *this; };

    ConstViewType getConstView() const { return *this; };

    template< typename OffsetsView,
              typename Fetch,
              typename Reduction,
              typename ResultKeeper,
              typename Real,
              typename... Args >
    void segmentsReduction( const OffsetsView& offsets,
                        Index first,
                        Index last,
                        Fetch& fetch,
                        const Reduction& reduction,
                        ResultKeeper& keeper,
                        const Real& zero,
                        Args... args ) const
    {
    }
};

template< typename Index,
          typename Device >
struct CSRAdaptiveKernel
{
    using IndexType = Index;
    using DeviceType = Device;
    using ViewType = CSRAdaptiveKernel< Index, Device >;
    using ConstViewType = CSRAdaptiveKernel< Index, Device >;

    template< typename Offsets >
    void init( const Offsets& offsets )
    {
        /*const Index rows = offsets.getSize();
        Index sum, start = 0, nextStart = 0;

        // Fill blocks
        std::vector<Block<Index>> inBlock;
        inBlock.reserve(rows);

        while (nextStart != rows - 1)
        {
            Type type;
            nextStart = findLimit<Real, Index, Device, KernelType>(
                start, *this, rows, type, sum );

            if (type == Type::LONG)
            {
                Index parts = roundUpDivision(sum, this->SHARED_PER_WARP);
                for (Index index = 0; index < parts; ++index)
                {
                    inBlock.emplace_back(start, Type::LONG, index);
                }
            }
            else
            {
                inBlock.emplace_back(start, type,
                    nextStart,
                    this->rowPointers.getElement(nextStart),
                    this->rowPointers.getElement(start) );
            }
            start = nextStart;
        }
        inBlock.emplace_back(nextStart);

        // Copy values
        this->blocks.setSize(inBlock.size());
        for (size_t i = 0; i < inBlock.size(); ++i)
            this->blocks.setElement(i, inBlock[i]);
        */
    };

    ViewType getView() { return view; };

    ConstViewType getConstView() const { return ConstViewType(); };

    template< typename OffsetsView,
              typename Fetch,
              typename Reduction,
              typename ResultKeeper,
              typename Real,
              typename... Args >
    void segmentsReduction( const OffsetsView& offsets,
                        Index first,
                        Index last,
                        Fetch& fetch,
                        const Reduction& reduction,
                        ResultKeeper& keeper,
                        const Real& zero,
                        Args... args ) const
    {
        view.segmentsReduction( offsets, first, last, fetch, reduction, keeper, zero, args... );
    }

    ViewType view;
};



      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL
