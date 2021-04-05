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

/**
 * \brief Data structure for CSR segments format.
 *
 * See \ref TNL::Algorithms::Segments for more details about segments.
 *
 * \tparam Device is type of device where the segments will be operating.
 * \tparam Index is type for indexing of the elements managed by the segments.
 * \tparam Kernel is type of kernel used for parallel operations with segments.
 *    It can be any of the following:
 *    \ref TNL::Containers::Segments::Kernels::CSRAdaptiveKernel,
 *    \ref TNL::Containers::Segments::Kernels::CSRHybridKernel,
 *    \ref TNL::Containers::Segments::Kernels::CSRScalarKernel,
 *    \ref TNL::Containers::Segments::Kernels::CSRVectorKernel
 *
 * \tparam IndexAllocator is allocator for supporting index containers.
 */
template< typename Device,
          typename Index,
          typename Kernel = CSRScalarKernel< Index, Device >,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
class CSR
{
   public:

      /**
       * \brief The device where the segments are operating.
       */
      using DeviceType = Device;

      /**
       * \brief The type used for indexing of segments elements.
       */
      using IndexType = std::remove_const_t< Index >;

      /**
       * \brief Type of kernel used for reduction operations.
       */
      using KernelType = Kernel;

      /**
       * \brief Type of container storing offsets of particular rows.
       */
      using OffsetsContainer = Containers::Vector< Index, DeviceType, IndexType, IndexAllocator >;

      /**
       * \brief Templated view type.
       *
       * \tparam Device_ is alternative device type for the view.
       * \tparam Index_ is alternative index type for the view.
       */
      template< typename Device_, typename Index_ >
      using ViewTemplate = CSRView< Device_, Index_, KernelType >;

      /**
       * \brief Type of segments view.1
       */
      using ViewType = CSRView< Device, Index, KernelType >;

      /**
       * \brief Type of constant segments view.
       */
      using ConstViewType = CSRView< Device, std::add_const_t< IndexType >, KernelType >;

      /**
       * \brief Accessor type fro one particular segment.
       */
      using SegmentViewType = SegmentView< IndexType, RowMajorOrder >;

      /**
       * \brief This functions says that CSR format is always organised in row-major order.
       */
      static constexpr ElementsOrganization getOrganization() { return RowMajorOrder; }

      /**
       * \brief This function says that CSR format does not use padding elements.
       */
      static constexpr bool havePadding() { return false; };

      /**
       * \brief Construct with no parameters to create empty segments.
       */
      CSR();

      /**
       * \brief Construct with segments sizes.
       *
       * The number of segments is given by the size of \e segmentsSizes. Particular elements
       * of this container define sizes of particular segments.
       *
       * \tparam SizesContainer is a type of container for segments sizes.
       * \param sizes is an instance of the container with the segments sizes.
       *
       * See the following example:
       *
       * \includelineno Algorithms/Segments/SegmentsExample_CSR_constructor_1.cpp
       *
       * The result looks as follows:
       *
       * \include SegmentsExample_CSR_constructor_1.out
       */
      template< typename SizesContainer >
      CSR( const SizesContainer& segmentsSizes );

      /**
       * \brief Construct with segments sizes in initializer list..
       *
       * The number of segments is given by the size of \e segmentsSizes. Particular elements
       * of this initializer list define sizes of particular segments.
       *
       * \tparam ListIndex is a type of indexes of the initializer list.
       * \param sizes is an instance of the container with the segments sizes.
       *
       * See the following example:
       *
       * \includelineno Algorithms/Segments/SegmentsExample_constructor_2.cpp
       *
       * The result looks as follows:
       *
       * \include SegmentsExample_constructor_1.out
       */
      template< typename ListIndex >
      CSR( const std::initializer_list< ListIndex >& segmentsSizes );

      /**
       * \brief Copy constructor.
       *
       * \param segments are the source segments.
       */
      CSR( const CSR& segments );

      /**
       * \brief Move constructor.
       *
       * \param segments  are the source segments.
       */
      CSR( const CSR&& segments );

      static String getSerializationType();

      static String getSegmentsType();

      /**
       * \brief Set sizes of particular segments.
       */
      template< typename SizesHolder >
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

      const OffsetsContainer& getOffsets() const;

      OffsetsContainer& getOffsets();

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
      void forAllSegments( Function&& f ) const;

      /***
       * \brief Go over all segments and perform a reduction in each of them.
       */
      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
      void reduceSegments( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero ) const;

      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
      void reduceAllSegments( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero ) const;

      CSR& operator=( const CSR& rhsSegments ) = default;

      template< typename Device_, typename Index_, typename Kernel_, typename IndexAllocator_ >
      CSR& operator=( const CSR< Device_, Index_, Kernel_, IndexAllocator_ >& source );

      void save( File& file ) const;

      void load( File& file );

   protected:

      OffsetsContainer offsets;

      KernelType kernel;
};

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
std::ostream& operator<<( std::ostream& str, const CSR< Device, Index, Kernel, IndexAllocator >& segments ) { return printSegments( segments, str ); }

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
