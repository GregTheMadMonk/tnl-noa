/***************************************************************************
                          ChunkedEllpack.hpp -  description
                             -------------------
    begin                : Jan 21, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Containers/Segments/ChunkedEllpack.h>
#include <TNL/Containers/Segments/Ellpack.h>

namespace TNL {
   namespace Containers {
      namespace Segments {


template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
ChunkedEllpack()
   : size( 0 ), alignedSize( 0 ), segmentsCount( 0 )
{
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
ChunkedEllpack( const Vector< IndexType, DeviceType, IndexType >& sizes )
   : size( 0 ), alignedSize( 0 ), segmentsCount( 0 )
{
   this->setSegmentsSizes( sizes );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
ChunkedEllpack( const ChunkedEllpack& slicedEllpack )
   : size( slicedEllpack.size ), alignedSize( slicedEllpack.alignedSize ),
     segmentsCount( slicedEllpack.segmentsCount ), sliceOffsets( slicedEllpack.sliceOffsets ),
     sliceSegmentSizes( slicedEllpack.sliceSegmentSizes )
{
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
ChunkedEllpack( const ChunkedEllpack&& slicedEllpack )
   : size( slicedEllpack.size ), alignedSize( slicedEllpack.alignedSize ),
     segmentsCount( slicedEllpack.segmentsCount ), sliceOffsets( slicedEllpack.sliceOffsets ),
     sliceSegmentSizes( slicedEllpack.sliceSegmentSizes )
{
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
String
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
getSerializationType()
{
   return "ChunkedEllpack< [any_device], " + TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
String
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
getSegmentsType()
{
   return ViewType::getSegmentsType();
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
typename ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::ViewType
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
getView()
{
   return ViewType( size, alignedSize, segmentsCount, sliceOffsets.getView(), sliceSegmentSizes.getView() );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
typename ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::ConstViewType
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
getConstView() const
{
   return ConstViewType( size, alignedSize, segmentsCount, sliceOffsets.getConstView(), sliceSegmentSizes.getConstView() );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
   template< typename SegmentsSizes >
void
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
resolveSliceSizes( SegmentsSizes& rowLengths )
{
   /****
    * Iterate over rows and allocate slices so that each slice has
    * approximately the same number of allocated elements
    */
   const IndexType desiredElementsInSlice =
            this->chunksInSlice * this->desiredChunkSize;

   IndexType row( 0 ),
             sliceSize( 0 ),
             allocatedElementsInSlice( 0 );
   numberOfSlices = 0;
   while( row < this->rows )
   {
      /****
       * Add one row to the current slice until we reach the desired
       * number of elements in a slice.
       */
      allocatedElementsInSlice += rowLengths[ row ];
      sliceSize++;
      row++;
      if( allocatedElementsInSlice < desiredElementsInSlice  )
          if( row < this->rows && sliceSize < chunksInSlice ) continue;
      TNL_ASSERT( sliceSize >0, );
      this->slices[ numberOfSlices ].size = sliceSize;
      this->slices[ numberOfSlices ].firstRow = row - sliceSize;
      this->slices[ numberOfSlices ].pointer = allocatedElementsInSlice; // this is only temporary
      sliceSize = 0;
      numberOfSlices++;
      allocatedElementsInSlice = 0;
   }
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
   template< typename SegmentsSizes >
bool
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
setSlice( SegmentsSizes& rowLengths,
          const IndexType sliceIndex,
          IndexType& elementsToAllocation )
{
   /****
    * Now, compute the number of chunks per each row.
    * Each row get one chunk by default.
    * Then each row will get additional chunks w.r. to the
    * number of the elements in the row. If there are some
    * free chunks left, repeat it again.
    */
   const IndexType sliceSize = this->slices[ sliceIndex ].size;
   const IndexType sliceBegin = this->slices[ sliceIndex ].firstRow;
   const IndexType allocatedElementsInSlice = this->slices[ sliceIndex ].pointer;
   const IndexType sliceEnd = sliceBegin + sliceSize;

   IndexType freeChunks = this->chunksInSlice - sliceSize;
   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
      this->rowToChunkMapping.setElement( i, 1 );

   int totalAddedChunks( 0 );
   int maxRowLength( rowLengths[ sliceBegin ] );
   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
   {
      double rowRatio( 0.0 );
      if( allocatedElementsInSlice != 0 )
         rowRatio = ( double ) rowLengths[ i ] / ( double ) allocatedElementsInSlice;
      const IndexType addedChunks = freeChunks * rowRatio;
      totalAddedChunks += addedChunks;
      this->rowToChunkMapping[ i ] += addedChunks;
      if( maxRowLength < rowLengths[ i ] )
         maxRowLength = rowLengths[ i ];
   }
   freeChunks -= totalAddedChunks;
   while( freeChunks )
      for( IndexType i = sliceBegin; i < sliceEnd && freeChunks; i++ )
         if( rowLengths[ i ] == maxRowLength )
         {
            this->rowToChunkMapping[ i ]++;
            freeChunks--;
         }

   /****
    * Compute the chunk size
    */
   IndexType maxChunkInSlice( 0 );
   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
   {
       maxChunkInSlice = max( maxChunkInSlice,
                          roundUpDivision( rowLengths[ i ], this->rowToChunkMapping[ i ] ) );
   }
      TNL_ASSERT( maxChunkInSlice > 0,
              std::cerr << " maxChunkInSlice = " << maxChunkInSlice << std::endl );

   /****
    * Set-up the slice info.
    */
   this->slices[ sliceIndex ].chunkSize = maxChunkInSlice;
   this->slices[ sliceIndex ].pointer = elementsToAllocation;
   elementsToAllocation += this->chunksInSlice * maxChunkInSlice;

   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
      this->rowToSliceMapping[ i ] = sliceIndex;

   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
   {
      this->rowPointers[ i + 1 ] = maxChunkInSlice*rowToChunkMapping[ i ];
      TNL_ASSERT( this->rowPointers[ i ] >= 0,
                 std::cerr << "this->rowPointers[ i ] = " << this->rowPointers[ i ] );
      TNL_ASSERT( this->rowPointers[ i + 1 ] >= 0,
                 std::cerr << "this->rowPointers[ i + 1 ] = " << this->rowPointers[ i + 1 ] );
   }

   /****
    * Finish the row to chunk mapping by computing the prefix sum.
    */
   for( IndexType j = sliceBegin + 1; j < sliceEnd; j++ )
      rowToChunkMapping[ j ] += rowToChunkMapping[ j - 1 ];
   return true;
}


template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int SliceSize >
   template< typename SizesHolder >
void
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
setSegmentsSizes( const SizesHolder& sizes )
{
      TNL_ASSERT_GT( this->getRows(), 0, "cannot set row lengths of an empty matrix" );
   TNL_ASSERT_GT( this->getColumns(), 0, "cannot set row lengths of an empty matrix" );
   TNL_ASSERT_EQ( this->getRows(), rowLengths.getSize(), "wrong size of the rowLengths vector" );

   IndexType elementsToAllocation( 0 );

   this->resolveSliceSizes( sizes );
   this->rowPointers.setElement( 0, 0 );
   for( IndexType sliceIndex = 0; sliceIndex < numberOfSlices; sliceIndex++ )
      this->setSlice( rowLengths, sliceIndex, elementsToAllocation );
   this->rowPointers.scan();

}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
Index
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
getSegmentsCount() const
{
   return this->segmentsCount;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
Index
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
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
          typename IndexAllocator,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
Index
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
getSize() const
{
   return this->size;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
Index
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
getStorageSize() const
{
   return this->alignedSize;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
Index
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
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
#ifdef __CUDA__ARCH__
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
          typename IndexAllocator,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
void
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
getSegmentAndLocalIndex( const Index globalIdx, Index& segmentIdx, Index& localIdx ) const
{
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
auto
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
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
          typename IndexAllocator,
          bool RowMajorOrder,
          int SliceSize >
   template< typename Function, typename... Args >
void
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
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
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx++  )
            if( ! f( segmentIdx, localIdx++, globalIdx, args... ) )
               break;
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
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SliceSize )
            if( ! f( segmentIdx, localIdx++, globalIdx, args... ) )
               break;
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l, args... );
   }
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int SliceSize >
   template< typename Function, typename... Args >
void
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
forAll( Function& f, Args... args ) const
{
   this->forSegments( 0, this->getSegmentsCount(), f, args... );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int SliceSize >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
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
         bool compute( true );
         IndexType localIdx( 0 );
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
         bool compute( true );
         IndexType localIdx( 0 );
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SliceSize  )
            reduction( aux, fetch( segmentIdx, localIdx++, globalIdx, compute, args... ) );
         keeper( segmentIdx, aux );
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l, args... );
   }
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int SliceSize >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
allReduction( Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->segmentsReduction( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int SliceSize >
   template< typename Device_, typename Index_, typename IndexAllocator_, bool RowMajorOrder_ >
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder, SliceSize >&
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
operator=( const ChunkedEllpack< Device_, Index_, IndexAllocator_, RowMajorOrder_, SliceSize >& source )
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
          bool RowMajorOrder,
          int SliceSize >
void
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
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
          bool RowMajorOrder,
          int SliceSize >
void
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
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
