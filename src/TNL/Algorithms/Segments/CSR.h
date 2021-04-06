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
       * \tparam SizesContainer is a type of container for segments sizes.  It can be \ref TNL::Containers::Array or
       *  \ref TNL::Containers::Vector for example.
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
       * \includelineno Algorithms/Segments/SegmentsExample_CSR_constructor_2.cpp
       *
       * The result looks as follows:
       *
       * \include SegmentsExample_CSR_constructor_2.out
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

      /**
       * \brief Returns string with serialization type.
       *
       * The string has a form `Algorithms::Segments::CSR< IndexType,  [any_device], [any_kernel], [any_allocator] >`.
       *
       * \return \ref String with the serialization type.
       *
       * \par Example
       * \include Algorithms/Segments/SegmentsExample_CSR_getSerializationType.cpp
       * \par Output
       * \include SegmentsExample_CSR_getSerializationType.out
       */
      static String getSerializationType();

      /**
       * \brief Returns string with segments type.
       *
       * The string has a form `CSR< KernelType >`.
       *
       * \return \ref String with the segments type.
       *
       * \par Example
       * \include Algorithms/Segments/SegmentsExample_CSR_getSegmentsType.cpp
       * \par Output
       * \include SegmentsExample_CSR_getSegmentsType.out
       */
      static String getSegmentsType();

      /**
       * \brief Set sizes of particular segments.
       *
       * \tparam SizesContainer is a container with segments sizes. It can be \ref TNL::Containers::Array or
       *  \ref TNL::Containers::Vector for example.
       *
       * \param segmentsSizes is an instance of the container with segments sizes.
       */
      template< typename SizesContainer >
      void setSegmentsSizes( const SizesContainer& segmentsSizes );

      /**
       * \brief Reset the segments to empty states.
       *
       * It means that there is no segment in the CSR segments.
       */
      void reset();

      /**
       * \brief Getter of a view object.
       *
       * \return View for this instance of CSR segments which can by used for example in
       *  lambda functions running in GPU kernels.
       */
      ViewType getView();

      /**
       * \brief Getter of a view object for constants instances.
       *
       * \return View for this instance of CSR segments which can by used for example in
       *  lambda functions running in GPU kernels.
       */
      const ConstViewType getConstView() const;

      /**
       * \brief Getter of number of segments.
       *
       * \return number of segments within this object.
       */
      __cuda_callable__
      IndexType getSegmentsCount() const;

      /**
       * \brief Returns size of particular segment.
       *
       * \return size of the segment number \e segmentIdx.
       */
      __cuda_callable__
      IndexType getSegmentSize( const IndexType segmentIdx ) const;

      /***
       * \brief Returns number of elements managed by all segments.
       *
       * \return number of elements managed by all segments.
       */
      __cuda_callable__
      IndexType getSize() const;

      /**
       * \brief Returns number of elements that needs to be allocated by a container connected to this segments.
       *
       * \return size of container connected to this segments.
       */
      __cuda_callable__
      IndexType getStorageSize() const;

      /**
       * \brief Computes the global index of an element managed by the segments.
       *
       * The global index serves as a refernce on the element in its container.
       *
       * \param segmentIdx is index of a segment with the element.
       * \param localIdx is tha local index of the element within the segment.
       * \return global index of the element.
       */
      __cuda_callable__
      IndexType getGlobalIndex( const Index segmentIdx, const Index localIdx ) const;

      /**
       * \brief Returns segment view (i.e. segment accessor) of segment with given index.
       *
       * \param segmentIdx is index of the request segment.
       * \return segment view of given segment.
       *
       * \par Example
       * \include Algorithms/Segments/SegmentsExample_CSR_getSegmentView.cpp
       * \par Output
       * \include SegmentsExample_CSR_getSegmentView.out
       */
      __cuda_callable__
      SegmentViewType getSegmentView( const IndexType segmentIdx ) const;

      /**
       * \brief Returns reference on constant vector with row offsets used in the CSR format.
       *
       * \return reference on constant vector with row offsets used in the CSR format.
       */
      const OffsetsContainer& getOffsets() const;

      /**
       * \brief Returns reference on vector with row offsets used in the CSR format.
       *
       * \return reference on vector with row offsets used in the CSR format.
       */
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

      template< typename Function >
      void sequentialForSegments( IndexType begin, IndexType end, Function&& f ) const;

      template< typename Function >
      void sequentialForAllSegments( Function&& f ) const;


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
