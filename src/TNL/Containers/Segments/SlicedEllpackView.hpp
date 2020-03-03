/***************************************************************************
                          SlicedEllpackView.hpp -  description
                             -------------------
    begin                : Dec 4, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Containers/Segments/SlicedEllpackView.h>

#include "SlicedEllpackView.h"

namespace TNL {
   namespace Containers {
      namespace Segments {


template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
SlicedEllpackView()
   : size( 0 ), alignedSize( 0 ), segmentsCount( 0 )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
SlicedEllpackView(  IndexType size,
                    IndexType alignedSize,
                    IndexType segmentsCount,
                    OffsetsView&& sliceOffsets,
                    OffsetsView&& sliceSegmentSizes )
   : size( size ), alignedSize( alignedSize ), segmentsCount( segmentsCount ),
     sliceOffsets( std::forward< OffsetsView >( sliceOffsets ) ), sliceSegmentSizes( std::forward< OffsetsView >( sliceSegmentSizes ) )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
SlicedEllpackView( const SlicedEllpackView& slicedEllpackView )
   : size( slicedEllpackView.size ), alignedSize( slicedEllpackView.alignedSize ),
     segmentsCount( slicedEllpackView.segmentsCount ), sliceOffsets( slicedEllpackView.sliceOffsets ),
     sliceSegmentSizes( slicedEllpackView.sliceSegmentSizes )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
SlicedEllpackView( const SlicedEllpackView&& slicedEllpackView )
   : size( slicedEllpackView.size ), alignedSize( slicedEllpackView.alignedSize ),
     segmentsCount( slicedEllpackView.segmentsCount ), sliceOffsets( slicedEllpackView.sliceOffsets ),
     sliceSegmentSizes( slicedEllpackView.sliceSegmentSizes )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
String
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
getSerializationType()
{
   return "SlicedEllpack< [any_device], " + TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
String
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
getSegmentsType()
{
   return "SlicedEllpack";
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
typename SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::ViewType
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
getView()
{
   return ViewType( size, alignedSize, segmentsCount, sliceOffsets, sliceSegmentSizes );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
typename SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::ConstViewType
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
getConstView() const
{
   return ConstViewType( size, alignedSize, segmentsCount, sliceOffsets.getConstView(), sliceSegmentSizes.getConstView() );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
Index
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
getSegmentsCount() const
{
   return this->segmentsCount;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
Index
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
getSegmentSize( const IndexType segmentIdx ) const
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
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
Index
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
getSize() const
{
   return this->size;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
Index
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
getStorageSize() const
{
   return this->alignedSize;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
Index
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
getGlobalIndex( const Index segmentIdx, const Index localIdx ) const
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
#ifdef __CUDA_ARCH__
      sliceOffset = this->sliceOffsets[ sliceIdx ];
      segmentSize = this->sliceSegmentSizes[ sliceIdx ];
#else
      sliceOffset = this->sliceOffsets.getElement( sliceIdx );
      segmentSize = this->sliceSegmentSizes.getElement( sliceIdx );
#endif
   }
   if( RowMajorOrder )
      return sliceOffset + segmentInSliceIdx * segmentSize + localIdx;
   else
      return sliceOffset + segmentInSliceIdx + SliceSize * localIdx;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
void
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
getSegmentAndLocalIndex( const Index globalIdx, Index& segmentIdx, Index& localIdx ) const
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
auto
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   const IndexType sliceIdx = segmentIdx / SliceSize;
   const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
   const IndexType& sliceOffset = this->sliceOffsets[ sliceIdx ];
   const IndexType& segmentSize = this->sliceSegmentSizes[ sliceIdx ];

   if( RowMajorOrder )
      return SegmentViewType( sliceOffset + segmentInSliceIdx * segmentSize, segmentSize, 1 );
   else
      return SegmentViewType( sliceOffset + segmentInSliceIdx, segmentSize, SliceSize );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
   template< typename Function, typename... Args >
void
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
forSegments( IndexType first, IndexType last, Function& f, Args... args ) const
{
   const auto sliceSegmentSizes_view = this->sliceSegmentSizes.getConstView();
   const auto sliceOffsets_view = this->sliceOffsets.getConstView();
   if( RowMajorOrder )
   {
      auto l = [=] __cuda_callable__ ( const IndexType segmentIdx, Args... args ) mutable {
         const IndexType sliceIdx = segmentIdx / SliceSize;
         const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
         const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
         const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx * segmentSize;
         const IndexType end = begin + segmentSize;
         IndexType localIdx( 0 );
         bool compute( true );
         for( IndexType globalIdx = begin; globalIdx < end && compute; globalIdx++  )
            f( segmentIdx, localIdx++, globalIdx, compute, args... );
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l, args... );
   }
   else
   {
      auto l = [=] __cuda_callable__ ( const IndexType segmentIdx, Args... args ) mutable {
         const IndexType sliceIdx = segmentIdx / SliceSize;
         const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
         const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
         const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx;
         const IndexType end = sliceOffsets_view[ sliceIdx + 1 ];
         IndexType localIdx( 0 );
         bool compute( true );
         for( IndexType globalIdx = begin; globalIdx < end && compute; globalIdx += SliceSize )
            f( segmentIdx, localIdx++, globalIdx, compute, args... );
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l, args... );
   }
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
   template< typename Function, typename... Args >
void
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
forAll( Function& f, Args... args ) const
{
   this->forSegments( 0, this->getSegmentsCount(), f, args... );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
segmentsReduction( IndexType first, IndexType last, Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   using RealType = decltype( fetch( IndexType(), IndexType(), IndexType(), std::declval< bool& >(), args... ) );
   const auto sliceSegmentSizes_view = this->sliceSegmentSizes.getConstView();
   const auto sliceOffsets_view = this->sliceOffsets.getConstView();
   if( RowMajorOrder )
   {
      auto l = [=] __cuda_callable__ ( const IndexType segmentIdx, Args... args ) mutable {
         const IndexType sliceIdx = segmentIdx / SliceSize;
         const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
         const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
         const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx * segmentSize;
         const IndexType end = begin + segmentSize;
         RealType aux( zero );
         IndexType localIdx( 0 );
         bool compute( true );
         for( IndexType globalIdx = begin; globalIdx< end; globalIdx++  )
            reduction( aux, fetch( segmentIdx, localIdx++, globalIdx, compute, args... ) );
         keeper( segmentIdx, aux );
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l, args... );
   }
   else
   {
      auto l = [=] __cuda_callable__ ( const IndexType segmentIdx, Args... args ) mutable {
         const IndexType sliceIdx = segmentIdx / SliceSize;
         const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
         const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
         const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx;
         const IndexType end = sliceOffsets_view[ sliceIdx + 1 ];
         RealType aux( zero );
         IndexType localIdx( 0 );
         bool compute( true );
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SliceSize  )
            reduction( aux, fetch( segmentIdx, localIdx++, globalIdx, compute, args... ) );
         keeper( segmentIdx, aux );
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l, args... );
   }
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
allReduction( Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->segmentsReduction( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >&
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
operator=( const SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >& view )
{
   this->size = view.size;
   this->alignedSize = view.alignedSize;
   this->segmentsCount = view.segmentsCount;
   this->sliceOffsets.bind( view.sliceOffsets );
   this->sliceSegmentSizes.bind( view.sliceSegmentSizes );
   return *this;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
void
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
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
          bool RowMajorOrder,
          int SliceSize >
void
SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >::
load( File& file )
{
   file.load( &size );
   file.load( &alignedSize );
   file.load( &segmentsCount );
   file >> this->sliceOffsets;
   file >> this->sliceSegmentSizes;
}

      } // namespace Segments
   }  // namespace Conatiners
} // namespace TNL
