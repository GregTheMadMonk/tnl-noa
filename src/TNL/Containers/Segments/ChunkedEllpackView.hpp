/***************************************************************************
                          ChunkedEllpackView.hpp -  description
                             -------------------
    begin                : Mar 21, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Containers/Segments/ChunkedEllpackView.h>
//#include <TNL/Containers/Segments/details/ChunkedEllpack.h>

namespace TNL {
   namespace Containers {
      namespace Segments {


template< typename Device,
          typename Index,
          bool RowMajorOrder >
__cuda_callable__
ChunkedEllpackView< Device, Index, RowMajorOrder >::
ChunkedEllpackView( const IndexType size,
                    const IndexType storageSize,
                    const IndexType chunksInSlice,
                    const IndexType desiredChunkSize,
                    const OffsetsView& rowToChunkMapping,
                    const OffsetsView& rowToSliceMapping,
                    const OffsetsView& rowPointers,
                    const ChunkedEllpackSliceInfoContainerView& slices,
                    const IndexType numberOfSlices )
: size( size ),
  storageSize( storageSize ),
  chunksInSlice( chunksInSlice ),
  desiredChunkSize( desiredChunkSize ),
  rowToChunkMapping( rowToChunkMapping ),
  rowToSliceMapping( rowToSliceMapping ),
  rowPointers( rowPointers ),
  slices( slices ),
  numberOfSlices( numberOfSlices )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
__cuda_callable__
ChunkedEllpackView< Device, Index, RowMajorOrder >::
ChunkedEllpackView( const IndexType size,
                    const IndexType storageSize,
                    const IndexType chunksInSlice,
                    const IndexType desiredChunkSize,
                    const OffsetsView&& rowToChunkMapping,
                    const OffsetsView&& rowToSliceMapping,
                    const OffsetsView&& rowPointers,
                    const ChunkedEllpackSliceInfoContainerView&& slices,
                    const IndexType numberOfSlices )
: size( size ),
  storageSize( storageSize ),
  chunksInSlice( chunksInSlice ),
  desiredChunkSize( desiredChunkSize ),
  rowToChunkMapping( rowToChunkMapping ),
  rowToSliceMapping( rowToSliceMapping ),
  rowPointers( rowPointers ),
  slices( slices ),
  numberOfSlices( numberOfSlices )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
__cuda_callable__
ChunkedEllpackView< Device, Index, RowMajorOrder >::
ChunkedEllpackView( const ChunkedEllpackView& chunked_ellpack_view )
: size( chunked_ellpack_view.size ),
  storageSize( chunked_ellpack_view.storageSize ),
  chunksInSlice( chunked_ellpack_view.chunksInSlice ),
  desiredChunkSize( chunked_ellpack_view.desiredChunkSize ),
  rowToChunkMapping( chunked_ellpack_view.rowToChunkMapping ),
  rowToSliceMapping( chunked_ellpack_view.rowToSliceMapping ),
  rowPointers( chunked_ellpack_view.rowPointers ),
  slices( chunked_ellpack_view.slices ),
  numberOfSlices( chunked_ellpack_view.numberOfSlices )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
__cuda_callable__
ChunkedEllpackView< Device, Index, RowMajorOrder >::
ChunkedEllpackView( const ChunkedEllpackView&& chunked_ellpack_view )
: size( chunked_ellpack_view.size ),
  storageSize( chunked_ellpack_view.storageSize ),
  chunksInSlice( chunked_ellpack_view.chunksInSlice ),
  desiredChunkSize( chunked_ellpack_view.desiredChunkSize ),
  rowToChunkMapping( std::move( chunked_ellpack_view.rowToChunkMapping ) ),
  rowToSliceMapping( std::move( chunked_ellpack_view.rowToSliceMapping ) ),
  rowPointers( std::move( chunked_ellpack_view.rowPointers ) ),
  slices( std::move( chunked_ellpack_view.slices ) ),
  numberOfSlices( chunked_ellpack_view.numberOfSlices )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
String
ChunkedEllpackView< Device, Index, RowMajorOrder >::
getSerializationType()
{
   return "ChunkedEllpack< [any_device], " + TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
String
ChunkedEllpackView< Device, Index, RowMajorOrder >::
getSegmentsType()
{
   return "ChunkedEllpack";
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
__cuda_callable__
typename ChunkedEllpackView< Device, Index, RowMajorOrder >::ViewType
ChunkedEllpackView< Device, Index, RowMajorOrder >::
getView()
{
   return ViewType( size, chunksInSlice, desiredChunkSize,
                    rowToChunkMapping.getView(),
                    rowToSliceMapping.getView(),
                    rowPointers.getView(),
                    slices.getView(),
                    numberOfSlices );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
__cuda_callable__
typename ChunkedEllpackView< Device, Index, RowMajorOrder >::ConstViewType
ChunkedEllpackView< Device, Index, RowMajorOrder >::
getConstView() const
{
   return ConstViewType( size, chunksInSlice, desiredChunkSize,
                         rowToChunkMapping.getConstView(),
                         rowToSliceMapping.getConstView(),
                         rowPointers.getConstView(),
                         slices.getConstView(),
                         numberOfSlices );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
__cuda_callable__
Index
ChunkedEllpackView< Device, Index, RowMajorOrder >::
getSegmentsCount() const
{
   return this->size;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
__cuda_callable__
Index
ChunkedEllpackView< Device, Index, RowMajorOrder >::
getSegmentSize( const IndexType segmentIdx ) const
{
   const IndexType& sliceIndex = rowToSliceMapping[ segmentIdx ];
   TNL_ASSERT_LE( sliceIndex, this->getSegmentsCount(), "" );
   IndexType firstChunkOfSegment( 0 );
   if( segmentIdx != slices[ sliceIndex ].firstSegment )
      firstChunkOfSegment = rowToChunkMapping[ segmentIdx - 1 ];

   const IndexType lastChunkOfSegment = rowToChunkMapping[ segmentIdx ];
   const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
   const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
   return chunkSize * segmentChunksCount;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
__cuda_callable__
Index
ChunkedEllpackView< Device, Index, RowMajorOrder >::
getSize() const
{
   return this->size;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
__cuda_callable__
Index
ChunkedEllpackView< Device, Index, RowMajorOrder >::
getStorageSize() const
{
   return this->storageSize;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
__cuda_callable__
Index
ChunkedEllpackView< Device, Index, RowMajorOrder >::
getGlobalIndex( const Index segmentIdx, const Index localIdx ) const
{
   const IndexType& sliceIndex = rowToSliceMapping[ segmentIdx ];
   TNL_ASSERT_LE( sliceIndex, this->size, "" );
   IndexType firstChunkOfSegment( 0 );
   if( segmentIdx != slices[ sliceIndex ].firstSegment )
      firstChunkOfSegment = rowToChunkMapping[ segmentIdx - 1 ];
   
   const IndexType lastChunkOfSegment = rowToChunkMapping[ segmentIdx ];
   const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
   const IndexType sliceOffset = slices[ sliceIndex ].pointer;
   const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
   TNL_ASSERT_LE( localIdx, segmentChunksCount * chunkSize, "" );

   if( RowMajorOrder )
      return sliceOffset + firstChunkOfSegment * chunkSize + localIdx;
   else
   {
      const IndexType inChunkOffset = localIdx % chunkSize;
      const IndexType chunkIdx = localIdx / chunkSize;
      return sliceOffset + inChunkOffset * chunksInSlice + firstChunkOfSegment + chunkIdx;
   }
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
__cuda_callable__
void
ChunkedEllpackView< Device, Index, RowMajorOrder >::
getSegmentAndLocalIndex( const Index globalIdx, Index& segmentIdx, Index& localIdx ) const
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
__cuda_callable__
auto
ChunkedEllpackView< Device, Index, RowMajorOrder >::
getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   const IndexType& sliceIndex = rowToSliceMapping[ segmentIdx ];
   TNL_ASSERT_LE( sliceIndex, this->size, "" );
   IndexType firstChunkOfSegment( 0 );
   if( segmentIdx != slices[ sliceIndex ].firstSegment )
      firstChunkOfSegment = rowToChunkMapping[ segmentIdx - 1 ];

   const IndexType lastChunkOfSegment = rowToChunkMapping[ segmentIdx ];
   const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
   const IndexType sliceOffset = slices[ sliceIndex ].pointer;
   const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
   const IndexType segmentSize = segmentChunksCount * chunkSize;

   if( RowMajorOrder )
      return SegmentViewType( sliceOffset + firstChunkOfSegment * chunkSize,
                              segmentSize,
                              chunkSize,
                              chunksInSlice );
   else // TODO FIX !!!!!!!!!!!!!!
      return SegmentViewType( sliceOffset + firstChunkOfSegment,
                              segmentSize,
                              chunkSize,
                              chunksInSlice );
   
   
   
   
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
   template< typename Function, typename... Args >
void
ChunkedEllpackView< Device, Index, RowMajorOrder >::
forSegments( IndexType first, IndexType last, Function& f, Args... args ) const
{
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      for( IndexType segmentIdx = first; segmentIdx < last; segmentIdx++ )
      {
         const IndexType& sliceIndex = rowToSliceMapping[ segmentIdx ];
         TNL_ASSERT_LE( sliceIndex, this->size, "" );
         IndexType firstChunkOfSegment( 0 );
         if( segmentIdx != slices[ sliceIndex ].firstSegment )
            firstChunkOfSegment = rowToChunkMapping[ segmentIdx - 1 ];

         const IndexType lastChunkOfSegment = rowToChunkMapping[ segmentIdx ];
         const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
         const IndexType sliceOffset = slices[ sliceIndex ].pointer;
         const IndexType chunkSize = slices[ sliceIndex ].chunkSize;

         const IndexType segmentSize = segmentChunksCount * chunkSize;
         bool compute( true );
         if( RowMajorOrder )
         {
            IndexType begin = sliceOffset + firstChunkOfSegment * chunkSize;
            IndexType end = begin + segmentSize;
            IndexType localIdx( 0 );
            for( IndexType j = begin; j < end && compute; j++ )
               f( segmentIdx, localIdx++, j, compute, args...);
         }
         else 
         {
            for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ )
            {
               IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
               IndexType end = begin + chunksInSlice * chunkSize;
               IndexType localIdx( 0 );
               for( IndexType j = begin; j < end && compute; j += chunksInSlice )
                  f( segmentIdx, localIdx++, j, compute, args...);
            }
         }
      }
   }

   /*const auto offsetsView = this->offsets;
   auto l = [=] __cuda_callable__ ( const IndexType segmentIdx, Args... args ) mutable {
      const IndexType begin = offsetsView[ segmentIdx ];
      const IndexType end = offsetsView[ segmentIdx + 1 ];
      IndexType localIdx( 0 );
      bool compute( true );
      for( IndexType globalIdx = begin; globalIdx < end && compute; globalIdx++  )
         f( segmentIdx, localIdx++, globalIdx, compute, args... );
   };
   Algorithms::ParallelFor< Device >::exec( first, last, l, args... );*/
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
   template< typename Function, typename... Args >
void
ChunkedEllpackView< Device, Index, RowMajorOrder >::
forAll( Function& f, Args... args ) const
{
   this->forSegments( 0, this->getSegmentsCount(), f, args... );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
ChunkedEllpackView< Device, Index, RowMajorOrder >::
segmentsReduction( IndexType first, IndexType last, Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   using RealType = decltype( fetch( IndexType(), IndexType(), IndexType(), std::declval< bool& >(), args... ) );
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      for( IndexType segmentIdx = first; segmentIdx < last; segmentIdx++ )
      {
         const IndexType& sliceIndex = rowToSliceMapping[ segmentIdx ];
         TNL_ASSERT_LE( sliceIndex, this->size, "" );
         IndexType firstChunkOfSegment( 0 );
         if( segmentIdx != slices[ sliceIndex ].firstSegment )
            firstChunkOfSegment = rowToChunkMapping[ segmentIdx - 1 ];

         const IndexType lastChunkOfSegment = rowToChunkMapping[ segmentIdx ];
         const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
         const IndexType sliceOffset = slices[ sliceIndex ].pointer;
         const IndexType chunkSize = slices[ sliceIndex ].chunkSize;

         const IndexType segmentSize = segmentChunksCount * chunkSize;
         RealType aux( zero );
         bool compute( true );
         if( RowMajorOrder )
         {
            IndexType begin = sliceOffset + firstChunkOfSegment * chunkSize;
            IndexType end = begin + segmentSize;
            IndexType localIdx( 0 );
            for( IndexType j = begin; j < end && compute; j++ )
               reduction( aux, fetch( segmentIdx, localIdx++, j, compute, args...) );
         }
         else
         {
            for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ )
            {
               IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
               IndexType end = begin + chunksInSlice * chunkSize;
               IndexType localIdx( 0 );
               for( IndexType j = begin; j < end && compute; j += chunksInSlice )
                  reduction( aux, fetch( segmentIdx, localIdx++, j, compute, args...) );
            }
         }
         keeper( segmentIdx, aux );
      }
   }
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
ChunkedEllpackView< Device, Index, RowMajorOrder >::
allReduction( Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->segmentsReduction( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
ChunkedEllpackView< Device, Index, RowMajorOrder >&
ChunkedEllpackView< Device, Index, RowMajorOrder >::
operator=( const ChunkedEllpackView& view )
{
   this->size = view.size;
   this->storageSize = view.storageSize;
   this->chunksInSlice = view.chunksInSlice;
   this->desiredChunkSize = view.desiredChunkSize;
   this->rowToChunkMapping.bind( view.rowToChunkMapping );
   this->rowToSliceMapping.bind( view.rowToSliceMapping );
   this->rowPointers.bind( view.rowPointers );
   this->slices.bind( view.slices );
   this->numberOfSlices = view.numberOfSlices;
   return *this;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
void
ChunkedEllpackView< Device, Index, RowMajorOrder >::
save( File& file ) const
{
   file.save( &this->size );
   file.save( &this->storageSize );
   file.save( &this->chunksInSlice );
   file.save( &this->desiredChunkSize );
   file << this->rowToChunkMapping
        << this->rowToSliceMapping
        << this->rowPointers
        << this->slices;
   file.save( &this->numberOfSlices );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
void
ChunkedEllpackView< Device, Index, RowMajorOrder >::
load( File& file )
{
   file.load( &this->size );
   file.load( &this->storageSize );
   file.load( &this->chunksInSlice );
   file.load( &this->desiredChunkSize );
   file >> this->rowToChunkMapping
        >> this->rowToSliceMapping
        >> this->rowPointers
        >> this->slices;
   file.load( &this->numberOfSlices );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder >
void
ChunkedEllpackView< Device, Index, RowMajorOrder >::
printStructure( std::ostream& str ) const
{
   //const IndexType numberOfSlices = this->getNumberOfSlices();
   str << "Segments count: " << this->getSize() << std::endl
       << "Slices: " << numberOfSlices << std::endl;
   for( IndexType i = 0; i < numberOfSlices; i++ )
      str << "   Slice " << i
          << " : size = " << this->slices.getElement( i ).size
          << " chunkSize = " << this->slices.getElement( i ).chunkSize
          << " firstSegment = " << this->slices.getElement( i ).firstSegment
          << " pointer = " << this->slices.getElement( i ).pointer << std::endl;
   for( IndexType i = 0; i < this->getSize(); i++ )
      str << "Segment " << i
          << " : slice = " << this->rowToSliceMapping.getElement( i )
          << " chunk = " << this->rowToChunkMapping.getElement( i ) << std::endl;
}

      } // namespace Segments
   }  // namespace Conatiners
} // namespace TNL
