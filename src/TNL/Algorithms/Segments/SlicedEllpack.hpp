/***************************************************************************
                          SlicedEllpack.hpp -  description
                             -------------------
    begin                : Dec 4, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/Segments/Ellpack.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {


template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
SlicedEllpack()
   : size( 0 ), alignedSize( 0 ), segmentsCount( 0 )
{
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
SlicedEllpack( const Containers::Vector< IndexType, DeviceType, IndexType >& sizes )
   : size( 0 ), alignedSize( 0 ), segmentsCount( 0 )
{
   this->setSegmentsSizes( sizes );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
SlicedEllpack( const SlicedEllpack& slicedEllpack )
   : size( slicedEllpack.size ), alignedSize( slicedEllpack.alignedSize ),
     segmentsCount( slicedEllpack.segmentsCount ), sliceOffsets( slicedEllpack.sliceOffsets ),
     sliceSegmentSizes( slicedEllpack.sliceSegmentSizes )
{
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
SlicedEllpack( const SlicedEllpack&& slicedEllpack )
   : size( slicedEllpack.size ), alignedSize( slicedEllpack.alignedSize ),
     segmentsCount( slicedEllpack.segmentsCount ), sliceOffsets( slicedEllpack.sliceOffsets ),
     sliceSegmentSizes( slicedEllpack.sliceSegmentSizes )
{
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
String
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
getSerializationType()
{
   return "SlicedEllpack< [any_device], " + TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
String
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
getSegmentsType()
{
   return ViewType::getSegmentsType();
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
typename SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::ViewType
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
getView()
{
   return ViewType( size, alignedSize, segmentsCount, sliceOffsets.getView(), sliceSegmentSizes.getView() );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
auto
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
getConstView() const -> const ConstViewType
{
   return ConstViewType( size, alignedSize, segmentsCount, sliceOffsets.getConstView(), sliceSegmentSizes.getConstView() );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
   template< typename SizesHolder >
void
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
setSegmentsSizes( const SizesHolder& sizes )
{
   this->segmentsCount = sizes.getSize();
   const IndexType slicesCount = roundUpDivision( this->segmentsCount, getSliceSize() );
   this->sliceOffsets.setSize( slicesCount + 1 );
   this->sliceOffsets = 0;
   this->sliceSegmentSizes.setSize( slicesCount );
   Ellpack< DeviceType, IndexType, IndexAllocator, RowMajorOrder > ellpack;
   ellpack.setSegmentsSizes( slicesCount, SliceSize );

   const IndexType _size = sizes.getSize();
   const auto sizes_view = sizes.getConstView();
   auto slices_view = this->sliceOffsets.getView();
   auto slice_segment_size_view = this->sliceSegmentSizes.getView();
   auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx, bool& compute ) -> IndexType {
      if( globalIdx < _size )
         return sizes_view[ globalIdx ];
      return 0;
   };
   auto reduce = [] __cuda_callable__ ( IndexType& aux, const IndexType i ) -> IndexType {
      return TNL::max( aux, i );
   };
   auto keep = [=] __cuda_callable__ ( IndexType i, IndexType res ) mutable {
      slices_view[ i ] = res * SliceSize;
      slice_segment_size_view[ i ] = res;
   };
   ellpack.allReduction( fetch, reduce, keep, std::numeric_limits< IndexType >::min() );
   inplaceExclusiveScan( this->sliceOffsets );
   this->size = sum( sizes );
   this->alignedSize = this->sliceOffsets.getElement( slicesCount );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
void
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
reset()
{
   this->size = 0;
   this->alignedSize = 0;
   this->segmentsCount = 0;
   this->sliceOffsets.reset();
   this->sliceSegmentSizes.reset();
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
__cuda_callable__ auto SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
getSegmentsCount() const -> IndexType
{
   return this->segmentsCount;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
__cuda_callable__ auto SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   const Index sliceIdx = segmentIdx / SliceSize;
   if( std::is_same< DeviceType, Devices::Host >::value )
      return this->sliceSegmentSizes[ sliceIdx ];
   else
   {
#ifdef __CUDA_ARCH__
   return this->sliceSegmentSizes[ sliceIdx ];
#else
   return this->sliceSegmentSizes.getElement( sliceIdx );
#endif
   }
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
__cuda_callable__ auto SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
getSize() const -> IndexType
{
   return this->size;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
__cuda_callable__ auto SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
getStorageSize() const -> IndexType
{
   return this->alignedSize;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
__cuda_callable__ auto SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
getGlobalIndex( const Index segmentIdx, const Index localIdx ) const -> IndexType
{
   const IndexType sliceIdx = segmentIdx / SliceSize;
   const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
   IndexType sliceOffset, segmentSize;
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      sliceOffset = this->sliceOffsets[ sliceIdx ];
      segmentSize = this->sliceSegmentSizes[ sliceIdx ];
   }
   else
   {
#ifdef __CUDA__ARCH__
      sliceOffset = this->sliceOffsets[ sliceIdx ];
      segmentSize = this->sliceSegmentSizes[ sliceIdx ];
#else
      sliceOffset = this->sliceOffsets.getElement( sliceIdx );
      segmentSize = this->sliceSegmentSizes.getElement( sliceIdx );
#endif
   }
   if( Organization == RowMajorOrder )
      return sliceOffset + segmentInSliceIdx * segmentSize + localIdx;
   else
      return sliceOffset + segmentInSliceIdx + SliceSize * localIdx;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
__cuda_callable__
auto
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   const IndexType sliceIdx = segmentIdx / SliceSize;
   const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
   const IndexType& sliceOffset = this->sliceOffsets[ sliceIdx ];
   const IndexType& segmentSize = this->sliceSegmentSizes[ sliceIdx ];

   if( Organization == RowMajorOrder )
      return SegmentViewType( segmentIdx, sliceOffset + segmentInSliceIdx * segmentSize, segmentSize, 1 );
   else
      return SegmentViewType( segmentIdx, sliceOffset + segmentInSliceIdx, segmentSize, SliceSize );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
   template< typename Function >
void
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
forElements( IndexType first, IndexType last, Function&& f ) const
{
   this->getConstView().forElements( first, last, f );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
   template< typename Function >
void
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
forAllElements( Function&& f ) const
{
   this->forElements( 0, this->getSegmentsCount(), f );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
   template< typename Function >
void
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
forSegments( IndexType begin, IndexType end, Function&& f ) const
{
   this->getConstView().forSegments( begin, end, f );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
   template< typename Function >
void
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
forAllSegments( Function&& f ) const
{
   this->getConstView().forAllSegments( f );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
reduceSegments( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->getConstView().reduceSegments( first, last, fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
allReduction( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->reduceSegments( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
   template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_ >
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >&
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
operator=( const SlicedEllpack< Device_, Index_, IndexAllocator_, Organization_, SliceSize >& source )
{
   this->size = source.size;
   this->alignedSize = source.alignedSize;
   this->segmentsCount = source.segmentsCount;
   this->sliceOffsets = source.sliceOffsets;
   this->sliceSegmentSizes = source.sliceSegmentSizes;
   return *this;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
void
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
save( File& file ) const
{
   file.save( &size );
   file.save( &alignedSize );
   file.save( &segmentsCount );
   file << this->sliceOffsets;
   file << this->sliceSegmentSizes;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int SliceSize >
void
SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize >::
load( File& file )
{
   file.load( &size );
   file.load( &alignedSize );
   file.load( &segmentsCount );
   file >> this->sliceOffsets;
   file >> this->sliceSegmentSizes;
}

      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL
