/***************************************************************************
                          CSRView.h -  description
                             -------------------
    begin                : Dec 11, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/SegmentView.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {

enum CSRKernelTypes { CSRScalarKernel, CSRVectorKernel, CSRLightKernel };

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ = CSRScalar >
class CSRView
{
   public:

      using DeviceType = Device;
      using IndexType = std::remove_const_t< Index >;
      using OffsetsView = typename Containers::VectorView< Index, DeviceType, IndexType >;
      using ConstOffsetsView = typename Containers::Vector< Index, DeviceType, IndexType >::ConstViewType;
      using ViewType = CSRView;
      template< typename Device_, typename Index_ >
      using ViewTemplate = CSRView< Device_, Index_ >;
      using ConstViewType = CSRView< Device, std::add_const_t< Index > >;
      using SegmentViewType = SegmentView< IndexType, RowMajorOrder >;
      CSRKernelTypes KernelType = KernelType_;

      __cuda_callable__
      CSRView();

      __cuda_callable__
      CSRView( const OffsetsView& offsets );

      __cuda_callable__
      CSRView( const OffsetsView&& offsets );

      __cuda_callable__
      CSRView( const CSRView& csr_view );

      __cuda_callable__
      CSRView( const CSRView&& csr_view );

      static String getSerializationType();

      static String getSegmentsType();

      __cuda_callable__
      ViewType getView();

      __cuda_callable__
      const ConstViewType getConstView() const;

      /**
       * \brief Number segments.
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

      /***
       * \brief Go over all segments and for each segment element call
       * function 'f' with arguments 'args'. The return type of 'f' is bool.
       * When its true, the for-loop continues. Once 'f' returns false, the for-loop
       * is terminated.
       */
      template< typename Function, typename... Args >
      void forSegments( IndexType first, IndexType last, Function& f, Args... args ) const;

      template< typename Function, typename... Args >
      void forAll( Function& f, Args... args ) const;


      /***
       * \brief Go over all segments and perform a reduction in each of them.
       */
      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
      void segmentsReduction( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const;

      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
      void allReduction( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const;

      CSRView& operator=( const CSRView& view );

      void save( File& file ) const;

      void load( File& file );

   protected:

      OffsetsView offsets;
};
      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL

#include <TNL/Algorithms/Segments/CSRView.hpp>
