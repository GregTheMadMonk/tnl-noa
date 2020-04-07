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
ChunkedEllpack( const Vector< IndexType, DeviceType, IndexType >& sizes )
{
   this->setSegmentsSizes( sizes );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
ChunkedEllpack( const ChunkedEllpack& chunkedEllpack )
   : size( chunkedEllpack.size ),
     storageSize( chunkedEllpack.storageSize ),
     chunksInSlice( chunkedEllpack.chunksInSlice ), 
     desiredChunkSize( chunkedEllpack.desiredChunkSize ),
     rowToChunkMapping( chunkedEllpack.rowToChunkMapping ),
     rowToSliceMapping( chunkedEllpack.rowToSliceMapping ),
     chunksToSegmentsMapping( chunkedEllpack. chunksToSegmentsMapping ),
     rowPointers( chunkedEllpack.rowPointers ),
     slices( chunkedEllpack.slices ),
     numberOfSlices( chunkedEllpack.numberOfSlices )
{
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
ChunkedEllpack( const ChunkedEllpack&& chunkedEllpack )
   : size( chunkedEllpack.size ),
     storageSize( chunkedEllpack.storageSize ),
     chunksInSlice( chunkedEllpack.chunksInSlice ),
     desiredChunkSize( chunkedEllpack.desiredChunkSize ),
     rowToChunkMapping( chunkedEllpack.rowToChunkMapping ),
     rowToSliceMapping( chunkedEllpack.rowTopSliceMapping ),
     chunksToSegmentsMapping( chunkedEllpack. chunksToSegmentsMapping ),
     rowPointers( chunkedEllpack.rowPointers ),
     slices( chunkedEllpack.slices ),
     numberOfSlices( chunkedEllpack.numberOfSlices )
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
   return ViewType( size, storageSize, chunksInSlice, desiredChunkSize,
                    rowToChunkMapping.getView(),
                    rowToSliceMapping.getView(),
                    chunksToSegmentsMapping.getView(),
                    rowPointers.getView(),
                    slices.getView(),
                    numberOfSlices );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
auto ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
getConstView() const -> const ConstViewType
{
   return ConstViewType( size, storageSize, chunksInSlice, desiredChunkSize,
                         rowToChunkMapping.getConstView(),
                         rowToSliceMapping.getConstView(),
                         chunksToSegmentsMapping.getConstView(),
                         rowPointers.getConstView(),
                         slices.getConstView(),
                         numberOfSlices );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
   template< typename SegmentsSizes >
void
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
resolveSliceSizes( SegmentsSizes& segmentsSizes )
{
   /****
    * Iterate over rows and allocate slices so that each slice has
    * approximately the same number of allocated elements
    */
   const IndexType desiredElementsInSlice =
            this->chunksInSlice * this->desiredChunkSize;

   IndexType segmentIdx( 0 ),
             sliceSize( 0 ),
             allocatedElementsInSlice( 0 );
   numberOfSlices = 0;
   while( segmentIdx < segmentsSizes.getSize() )
   {
      /****
       * Add one row to the current slice until we reach the desired
       * number of elements in a slice.
       */
      allocatedElementsInSlice += segmentsSizes[ segmentIdx ];
      sliceSize++;
      segmentIdx++;
      if( allocatedElementsInSlice < desiredElementsInSlice  )
          if( segmentIdx < segmentsSizes.getSize() && sliceSize < chunksInSlice ) continue;
      TNL_ASSERT( sliceSize >0, );
      this->slices[ numberOfSlices ].size = sliceSize;
      this->slices[ numberOfSlices ].firstSegment = segmentIdx - sliceSize;
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
   const IndexType sliceBegin = this->slices[ sliceIndex ].firstSegment;
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
      TNL_ASSERT_NE( this->rowToChunkMapping[ i ], 0, "" );
      maxChunkInSlice = TNL::max( maxChunkInSlice,
                              roundUpDivision( rowLengths[ i ], this->rowToChunkMapping[ i ] ) );
   }
   TNL_ASSERT_GT( maxChunkInSlice, 0, "" );

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
          bool RowMajorOrder >
   template< typename SizesHolder >
void
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
setSegmentsSizes( const SizesHolder& segmentsSizes )
{
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      this->size = segmentsSizes.getSize();
      this->slices.setSize( this->size );
      this->rowToChunkMapping.setSize( this->size );
      this->rowToSliceMapping.setSize( this->size );
      this->rowPointers.setSize( this->size + 1 );

      this->resolveSliceSizes( segmentsSizes );
      this->rowPointers.setElement( 0, 0 );
      this->storageSize = 0;
      for( IndexType sliceIndex = 0; sliceIndex < numberOfSlices; sliceIndex++ )
         this->setSlice( segmentsSizes, sliceIndex, storageSize );
      this->rowPointers.scan();
      IndexType chunksCount = this->numberOfSlices * this->chunksInSlice;
      this->chunksToSegmentsMapping.setSize( chunksCount );
      IndexType chunkIdx( 0 );
      for( IndexType segmentIdx = 0; segmentIdx < this->size; segmentIdx++ )
      {
         const IndexType& sliceIdx = rowToSliceMapping[ segmentIdx ];
         IndexType firstChunkOfSegment( 0 );
         if( segmentIdx != slices[ sliceIdx ].firstSegment )
               firstChunkOfSegment = rowToChunkMapping[ segmentIdx - 1 ];

         const IndexType lastChunkOfSegment = rowToChunkMapping[ segmentIdx ];
         const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
         for( IndexType i = 0; i < segmentChunksCount; i++ )
            this->chunksToSegmentsMapping[ chunkIdx++ ] = segmentIdx;
      }
   }
   else
   {
      ChunkedEllpack< Devices::Host, Index, typename Allocators::Default< Devices::Host >::template Allocator< Index >, RowMajorOrder > hostSegments;
      Containers::Vector< IndexType, Devices::Host, IndexType > hostSegmentsSizes( segmentsSizes );
      hostSegments.setSegmentsSizes( hostSegmentsSizes );
      *this = hostSegments;
   }
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
__cuda_callable__ auto ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
getSegmentsCount() const -> IndexType
{
   return this->segmentsCount;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
auto ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   return details::ChunkedEllpack< IndexType, DeviceType, RowMajorOrder >::getSegmentSize(
      rowToSliceMapping.getView(),
      slices.getView(),
      rowToChunkMapping.getView(),
      segmentIdx );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
__cuda_callable__ auto ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
getSize() const -> IndexType
{
   return this->size;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
__cuda_callable__ auto ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
getStorageSize() const -> IndexType
{
   return this->storageSize;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
__cuda_callable__ auto ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
getGlobalIndex( const Index segmentIdx, const Index localIdx ) const -> IndexType
{
      return details::ChunkedEllpack< IndexType, DeviceType, RowMajorOrder >::getGlobalIndex(
         rowToSliceMapping,
         slices,
         rowToChunkMapping,
         chunksInSlice,
         segmentIdx,
         localIdx );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
__cuda_callable__ auto ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
   template< typename Function, typename... Args >
void
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
forSegments( IndexType first, IndexType last, Function& f, Args... args ) const
{
   this->getConstView().forSegments( first, last, f, args... );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
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
          bool RowMajorOrder >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
segmentsReduction( IndexType first, IndexType last, Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->getConstView().segmentsReduction( first, last, fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
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
          bool RowMajorOrder >
   template< typename Device_, typename Index_, typename IndexAllocator_, bool RowMajorOrder_ >
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >&
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
operator=( const ChunkedEllpack< Device_, Index_, IndexAllocator_, RowMajorOrder_ >& source )
{
   this->size = source.size;
   this->storageSize = source.storageSize;
   this->chunksInSlice = source.chunksInSlice;
   this->desiredChunkSize = source.desiredChunkSize;
   this->rowToChunkMapping = source.rowToChunkMapping;
   this->rowToSliceMapping = source.rowToSliceMapping;
   this->rowPointers = source.rowPointers;
   this->chunksToSegmentsMapping = source.chunksToSegmentsMapping;
   this->slices = source.slices;
   this->numberOfSlices = source.numberOfSlices;
   return *this;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
void
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
save( File& file ) const
{
   file.save( &this->size );
   file.save( &this->storageSize );
   file.save( &this->chunksInSlice );
   file.save( &this->desiredChunkSize );
   file << this->rowToChunkMapping
        << this->rowToSliceMapping
        << this->rowPointers
        << this->chunksToSegmentsMapping
        << this->slices;
   file.save( this->numberOfSlices );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
void
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
load( File& file )
{
   file.load( &this->size );
   file.load( &this->storageSize );
   file.load( &this->chunksInSlice );
   file.load( &this->desiredChunkSize );
   file >> this->rowToChunkMapping
        >> this->rowToSliceMapping
        >> this->chunksToSegmentsMapping
        >> this->rowPointers
        >> this->slices;
   file.load( &this->numberOfSlices );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder >
void
ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >::
printStructure( std::ostream& str )
{
   this->getView().printStructure( str );
}

      } // namespace Segments
   }  // namespace Conatiners
} // namespace TNL
