/***************************************************************************
                          CSR.h -  description
                             -------------------
    begin                : Nov 29, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/CSRView.h>
#include <TNL/Algorithms/Segments/SegmentView.h>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {

template< typename Device,
          typename Index,
          typename Kernel = CSRScalarKernel< Index, Device >,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
class CSR
{
   public:

      using DeviceType = Device;
      using IndexType = std::remove_const_t< Index >;
      using KernelType = Kernel;
      using OffsetsHolder = Containers::Vector< Index, DeviceType, IndexType, IndexAllocator >;
      using SegmentsSizes = OffsetsHolder;
      template< typename Device_, typename Index_ >
      using ViewTemplate = CSRView< Device_, Index_, KernelType >;
      using ViewType = CSRView< Device, Index, KernelType >;
      using ConstViewType = CSRView< Device, std::add_const_t< IndexType >, KernelType >;
      using SegmentViewType = SegmentView< IndexType, RowMajorOrder >;

      static constexpr ElementsOrganization getOrganization() { return ColumnMajorOrder; }

      static constexpr bool havePadding() { return false; };

      CSR();

      CSR( const SegmentsSizes& sizes );

      CSR( const CSR& segments );

      CSR( const CSR&& segments );

      static String getSerializationType();

      static String getSegmentsType();

      /**
       * \brief Set sizes of particular segments.
       */
      template< typename SizesHolder = OffsetsHolder >
      void setSegmentsSizes( const SizesHolder& sizes );

      void reset();

      ViewType getView();

      const ConstViewType getConstView() const;

      /**
       * \brief Number of segments.
       */
      __cuda_callable__
      IndexType getSegmentsCount() const;

      /***
       * \brief Returns size of the segment number \r segmentIdx
       */
      __cuda_callable__
      IndexType getSegmentSize( const IndexType segmentIdx ) const;

      /***
       * \brief Returns number of elements managed by all segments.
       */
      __cuda_callable__
      IndexType getSize() const;

      /***
       * \brief Returns number of elements that needs to be allocated.
       */
      __cuda_callable__
      IndexType getStorageSize() const;

      __cuda_callable__
      IndexType getGlobalIndex( const Index segmentIdx, const Index localIdx ) const;

      __cuda_callable__
      SegmentViewType getSegmentView( const IndexType segmentIdx ) const;

      const OffsetsHolder& getOffsets() const;

      OffsetsHolder& getOffsets();

      /***
       * \brief Go over all segments and for each segment element call
       * function 'f'. The return type of 'f' is bool.
       * When its true, the for-loop continues. Once 'f' returns false, the for-loop
       * is terminated.
       */
      template< typename Function >
      void forElements( IndexType begin, IndexType end, Function&& f ) const;

      template< typename Function >
      void forAllElements( Function&& f ) const;

      template< typename Function >
      void forSegments( IndexType begin, IndexType end, Function&& f ) const;

      template< typename Function >
      void forEachSegment( Function&& f ) const;

      /***
       * \brief Go over all segments and perform a reduction in each of them.
       */
      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
      void segmentsReduction( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const;

      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
      void allReduction( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const;

      CSR& operator=( const CSR& rhsSegments ) = default;

      template< typename Device_, typename Index_, typename Kernel_, typename IndexAllocator_ >
      CSR& operator=( const CSR< Device_, Index_, Kernel_, IndexAllocator_ >& source );

      void save( File& file ) const;

      void load( File& file );

   protected:

      OffsetsHolder offsets;

      KernelType kernel;
};

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
using CSRScalar = CSR< Device, Index, CSRScalarKernel< Index, Device >, IndexAllocator >;

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
using CSRVector = CSR< Device, Index, CSRVectorKernel< Index, Device >, IndexAllocator >;

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
using CSRHybrid = CSR< Device, Index, CSRHybridKernel< Index, Device >, IndexAllocator >;

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
using CSRAdaptive = CSR< Device, Index, CSRAdaptiveKernel< Index, Device >, IndexAllocator >;

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
using CSRDefault = CSRScalar< Device, Index, IndexAllocator >;


      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL

#include <TNL/Algorithms/Segments/CSR.hpp>
