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
#include <TNL/Algorithms/Segments/ChunkedEllpackView.h>
#include <TNL/Algorithms/Segments/details/LambdaAdapter.h>
//#include <TNL/Algorithms/Segments/details/ChunkedEllpack.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
ChunkedEllpackView< Device, Index, Organization >::
ChunkedEllpackView( const IndexType size,
                    const IndexType storageSize,
                    const IndexType chunksInSlice,
                    const IndexType desiredChunkSize,
                    const OffsetsView& rowToChunkMapping,
                    const OffsetsView& rowToSliceMapping,
                    const OffsetsView& chunksToSegmentsMapping,
                    const OffsetsView& rowPointers,
                    const ChunkedEllpackSliceInfoContainerView& slices,
                    const IndexType numberOfSlices )
: size( size ),
  storageSize( storageSize ),
  numberOfSlices( numberOfSlices ),
  chunksInSlice( chunksInSlice ),
  desiredChunkSize( desiredChunkSize ),
  rowToSliceMapping( rowToSliceMapping ),
  rowToChunkMapping( rowToChunkMapping ),
  chunksToSegmentsMapping( chunksToSegmentsMapping ),
  rowPointers( rowPointers ),
  slices( slices )
{
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
ChunkedEllpackView< Device, Index, Organization >::
ChunkedEllpackView( const IndexType size,
                    const IndexType storageSize,
                    const IndexType chunksInSlice,
                    const IndexType desiredChunkSize,
                    const OffsetsView&& rowToChunkMapping,
                    const OffsetsView&& rowToSliceMapping,
                    const OffsetsView&& chunksToSegmentsMapping,
                    const OffsetsView&& rowPointers,
                    const ChunkedEllpackSliceInfoContainerView&& slices,
                    const IndexType numberOfSlices )
: size( size ),
  storageSize( storageSize ),
  numberOfSlices( numberOfSlices ),
  chunksInSlice( chunksInSlice ),
  desiredChunkSize( desiredChunkSize ),
  rowToSliceMapping( std::move( rowToSliceMapping ) ),
  rowToChunkMapping( std::move( rowToChunkMapping ) ),
  chunksToSegmentsMapping( std::move( chunksToSegmentsMapping ) ),
  rowPointers( std::move( rowPointers ) ),
  slices( std::move( slices ) )
{
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
ChunkedEllpackView< Device, Index, Organization >::
ChunkedEllpackView( const ChunkedEllpackView& chunked_ellpack_view )
: size( chunked_ellpack_view.size ),
  storageSize( chunked_ellpack_view.storageSize ),
  numberOfSlices( chunked_ellpack_view.numberOfSlices ),
  chunksInSlice( chunked_ellpack_view.chunksInSlice ),
  desiredChunkSize( chunked_ellpack_view.desiredChunkSize ),
  rowToSliceMapping( chunked_ellpack_view.rowToSliceMapping ),
  rowToChunkMapping( chunked_ellpack_view.rowToChunkMapping ),
  chunksToSegmentsMapping( chunked_ellpack_view.chunksToSegmentsMapping ),
  rowPointers( chunked_ellpack_view.rowPointers ),
  slices( chunked_ellpack_view.slices )
{
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
ChunkedEllpackView< Device, Index, Organization >::
ChunkedEllpackView( const ChunkedEllpackView&& chunked_ellpack_view )
: size( chunked_ellpack_view.size ),
  storageSize( chunked_ellpack_view.storageSize ),
  numberOfSlices( chunked_ellpack_view.numberOfSlices ),
  chunksInSlice( chunked_ellpack_view.chunksInSlice ),
  desiredChunkSize( chunked_ellpack_view.desiredChunkSize ),
  rowToSliceMapping( std::move( chunked_ellpack_view.rowToSliceMapping ) ),
  rowToChunkMapping( std::move( chunked_ellpack_view.rowToChunkMapping ) ),
  chunksToSegmentsMapping( std::move( chunked_ellpack_view.chunksToSegmentsMapping ) ),
  rowPointers( std::move( chunked_ellpack_view.rowPointers ) ),
  slices( std::move( chunked_ellpack_view.slices ) )
{
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
String
ChunkedEllpackView< Device, Index, Organization >::
getSerializationType()
{
   return "ChunkedEllpack< [any_device], " + TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
String
ChunkedEllpackView< Device, Index, Organization >::
getSegmentsType()
{
   return "ChunkedEllpack";
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
typename ChunkedEllpackView< Device, Index, Organization >::ViewType
ChunkedEllpackView< Device, Index, Organization >::
getView()
{
   return ViewType( size, chunksInSlice, desiredChunkSize,
                    rowToChunkMapping.getView(),
                    rowToSliceMapping.getView(),
                    chunksToSegmentsMapping.getView(),
                    rowPointers.getView(),
                    slices.getView(),
                    numberOfSlices );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__ auto ChunkedEllpackView< Device, Index, Organization >::
getConstView() const -> const ConstViewType
{
   return ConstViewType( size, chunksInSlice, desiredChunkSize,
                         rowToChunkMapping.getConstView(),
                         rowToSliceMapping.getConstView(),
                         chunksToSegmentsMapping.getConstView(),
                         rowPointers.getConstView(),
                         slices.getConstView(),
                         numberOfSlices );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__ auto ChunkedEllpackView< Device, Index, Organization >::
getSegmentsCount() const -> IndexType
{
   return this->size;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__ auto ChunkedEllpackView< Device, Index, Organization >::
getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   if( std::is_same< DeviceType, Devices::Host >::value )
      return details::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentSizeDirect(
         rowToSliceMapping,
         slices,
         rowToChunkMapping,
         segmentIdx );
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef __CUDA_ARCH__
      return details::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentSizeDirect(
         rowToSliceMapping,
         slices,
         rowToChunkMapping,
         segmentIdx );
#else
      return details::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentSize(
         rowToSliceMapping,
         slices,
         rowToChunkMapping,
         segmentIdx );
#endif
   }
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__ auto ChunkedEllpackView< Device, Index, Organization >::
getSize() const -> IndexType
{
   return this->size;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__ auto ChunkedEllpackView< Device, Index, Organization >::
getStorageSize() const -> IndexType
{
   return this->storageSize;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__ auto ChunkedEllpackView< Device, Index, Organization >::
getGlobalIndex( const Index segmentIdx, const Index localIdx ) const -> IndexType
{
   if( std::is_same< DeviceType, Devices::Host >::value )
      return details::ChunkedEllpack< IndexType, DeviceType, Organization >::getGlobalIndexDirect(
         rowToSliceMapping,
         slices,
         rowToChunkMapping,
         chunksInSlice,
         segmentIdx,
         localIdx );
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef __CUDA_ARCH__
      return details::ChunkedEllpack< IndexType, DeviceType, Organization >::getGlobalIndexDirect(
         rowToSliceMapping,
         slices,
         rowToChunkMapping,
         chunksInSlice,
         segmentIdx,
         localIdx );
#else
      return details::ChunkedEllpack< IndexType, DeviceType, Organization >::getGlobalIndex(
         rowToSliceMapping,
         slices,
         rowToChunkMapping,
         chunksInSlice,
         segmentIdx,
         localIdx );
#endif
   }
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackView< Device, Index, Organization >::
getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   if( std::is_same< DeviceType, Devices::Host >::value )
      return details::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentViewDirect(
         rowToSliceMapping,
         slices,
         rowToChunkMapping,
         chunksInSlice,
         segmentIdx );
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef __CUDA_ARCH__
      return details::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentViewDirect(
         rowToSliceMapping,
         slices,
         rowToChunkMapping,
         chunksInSlice,
         segmentIdx );
#else
      return details::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentView(
         rowToSliceMapping,
         slices,
         rowToChunkMapping,
         chunksInSlice,
         segmentIdx );
#endif
   }
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Function, typename... Args >
void
ChunkedEllpackView< Device, Index, Organization >::
forElements( IndexType first, IndexType last, Function& f, Args... args ) const
{
   const IndexType chunksInSlice = this->chunksInSlice;
   auto rowToChunkMapping = this->rowToChunkMapping;
   auto rowToSliceMapping = this->rowToSliceMapping;
   auto slices = this->slices;
   auto work = [=] __cuda_callable__ ( IndexType segmentIdx, Args... args ) mutable {
      const IndexType sliceIdx = rowToSliceMapping[ segmentIdx ];

      IndexType firstChunkOfSegment( 0 );
      if( segmentIdx != slices[ sliceIdx ].firstSegment )
      {
         firstChunkOfSegment = rowToChunkMapping[ segmentIdx - 1 ];
      }

      const IndexType lastChunkOfSegment = rowToChunkMapping[ segmentIdx ];
      const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
      const IndexType sliceOffset = slices[ sliceIdx ].pointer;
      const IndexType chunkSize = slices[ sliceIdx ].chunkSize;

      const IndexType segmentSize = segmentChunksCount * chunkSize;
      bool compute( true );
      if( Organization == RowMajorOrder )
      {
         IndexType begin = sliceOffset + firstChunkOfSegment * chunkSize;
         IndexType end = begin + segmentSize;
         IndexType localIdx( 0 );
         for( IndexType j = begin; j < end && compute; j++ )
            f( segmentIdx, localIdx++, j, compute, args...);
      }
      else
      {
         IndexType localIdx( 0 );
         for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ )
         {
            IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
            IndexType end = begin + chunksInSlice * chunkSize;
            for( IndexType j = begin; j < end && compute; j += chunksInSlice )
            {
               f( segmentIdx, localIdx++, j, compute, args...);
            }
         }
      }
   };
   Algorithms::ParallelFor< DeviceType >::exec( first, last , work, args... );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Function, typename... Args >
void
ChunkedEllpackView< Device, Index, Organization >::
forEachElement( Function& f, Args... args ) const
{
   this->forElements( 0, this->getSegmentsCount(), f, args... );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
ChunkedEllpackView< Device, Index, Organization >::
segmentsReduction( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   using RealType = typename details::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      //segmentsReductionKernel( 0, first, last, fetch, reduction, keeper, zero, args... );
      //return;
      
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
         if( Organization == RowMajorOrder )
         {
            IndexType begin = sliceOffset + firstChunkOfSegment * chunkSize;
            IndexType end = begin + segmentSize;
            IndexType localIdx( 0 );
            for( IndexType globalIdx = begin; globalIdx < end && compute; globalIdx++ )
               aux = reduction( aux, details::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx, compute ) );
         }
         else
         {
            for( IndexType chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ )
            {
               IndexType begin = sliceOffset + firstChunkOfSegment + chunkIdx;
               IndexType end = begin + chunksInSlice * chunkSize;
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end && compute; globalIdx += chunksInSlice )
                  aux = reduction( aux, details::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx, compute ) );
            }
         }
         keeper( segmentIdx, aux );
      }
   }
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      //const IndexType chunksCount = this->numberOfSlices * this->chunksInSlice;
      // TODO: This ignores parameters first and last
      const IndexType cudaBlocks = this->numberOfSlices;
      const IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
      dim3 cudaBlockSize( this->chunksInSlice ), cudaGridSize;
      const IndexType sharedMemory = cudaBlockSize.x * sizeof( RealType );

      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
      {
         if( gridIdx == cudaGrids - 1 )
            cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
         details::ChunkedEllpackSegmentsReductionKernel< ViewType, IndexType, Fetch, Reduction, ResultKeeper, Real, Args...  >
            <<< cudaGridSize, cudaBlockSize, sharedMemory  >>>
            ( *this, gridIdx, first, last, fetch, reduction, keeper, zero, args... );
      }
#endif
   }
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
ChunkedEllpackView< Device, Index, Organization >::
allReduction( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->segmentsReduction( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
ChunkedEllpackView< Device, Index, Organization >&
ChunkedEllpackView< Device, Index, Organization >::
operator=( const ChunkedEllpackView& view )
{
   this->size = view.size;
   this->storageSize = view.storageSize;
   this->chunksInSlice = view.chunksInSlice;
   this->desiredChunkSize = view.desiredChunkSize;
   this->rowToChunkMapping.bind( view.rowToChunkMapping );
   this->chunksToSegmentsMapping.bind( view.chunksToSegmentsMapping );
   this->rowToSliceMapping.bind( view.rowToSliceMapping );
   this->rowPointers.bind( view.rowPointers );
   this->slices.bind( view.slices );
   this->numberOfSlices = view.numberOfSlices;
   return *this;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
void
ChunkedEllpackView< Device, Index, Organization >::
save( File& file ) const
{
   file.save( &this->size );
   file.save( &this->storageSize );
   file.save( &this->chunksInSlice );
   file.save( &this->desiredChunkSize );
   file << this->rowToChunkMapping
        << this->chunksToSegmentsMapping
        << this->rowToSliceMapping
        << this->rowPointers
        << this->slices;
   file.save( &this->numberOfSlices );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
void
ChunkedEllpackView< Device, Index, Organization >::
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

#ifdef HAVE_CUDA
template< typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Real,
             typename... Args >
__device__
void
ChunkedEllpackView< Device, Index, Organization >::
segmentsReductionKernelWithAllParameters( IndexType gridIdx,
                                          IndexType first,
                                          IndexType last,
                                          Fetch fetch,
                                          Reduction reduction,
                                          ResultKeeper keeper,
                                          Real zero,
                                          Args... args ) const
{
   using RealType = decltype( fetch( IndexType(), IndexType(), IndexType(), std::declval< bool& >(), args... ) );

   const IndexType firstSlice = rowToSliceMapping[ first ];
   const IndexType lastSlice = rowToSliceMapping[ last - 1 ];

   const IndexType sliceIdx = firstSlice + gridIdx * Cuda::getMaxGridSize() + blockIdx.x;
   if( sliceIdx > lastSlice )
      return;

   RealType* chunksResults = Cuda::getSharedMemory< RealType >();
   __shared__ details::ChunkedEllpackSliceInfo< IndexType > sliceInfo;
   if( threadIdx.x == 0 )
      sliceInfo = this->slices[ sliceIdx ];
   chunksResults[ threadIdx.x ] = zero;
   __syncthreads();



   const IndexType sliceOffset = sliceInfo.pointer;
   const IndexType chunkSize = sliceInfo.chunkSize;
   const IndexType chunkIdx = sliceIdx * chunksInSlice + threadIdx.x;
   const IndexType segmentIdx = this->chunksToSegmentsMapping[ chunkIdx ];
   IndexType firstChunkOfSegment( 0 );
   if( segmentIdx != sliceInfo.firstSegment )
      firstChunkOfSegment = rowToChunkMapping[ segmentIdx - 1 ];
   IndexType localIdx = ( threadIdx.x - firstChunkOfSegment ) * chunkSize;
   bool compute( true );

   if( Organization == RowMajorOrder )
   {
      IndexType begin = sliceOffset + threadIdx.x * chunkSize; // threadIdx.x = chunkIdx within the slice
      IndexType end = begin + chunkSize;
      for( IndexType j = begin; j < end && compute; j++ )
         chunksResults[ threadIdx.x ] = reduction( chunksResults[ threadIdx.x ], fetch( segmentIdx, localIdx++, j, compute ) );
   }
   else
   {
      const IndexType begin = sliceOffset + threadIdx.x; // threadIdx.x = chunkIdx within the slice
      const IndexType end = begin + chunksInSlice * chunkSize;
         for( IndexType j = begin; j < end && compute; j += chunksInSlice )
            chunksResults[ threadIdx.x ] = reduction( chunksResults[ threadIdx.x ], fetch( segmentIdx, localIdx++, j, compute ) );
   }
   __syncthreads();
   if( threadIdx.x < sliceInfo.size )
   {
      const IndexType row = sliceInfo.firstSegment + threadIdx.x;
      IndexType chunkIndex( 0 );
      if( threadIdx.x != 0 )
         chunkIndex = this->rowToChunkMapping[ row - 1 ];
      const IndexType lastChunk = this->rowToChunkMapping[ row ];
      RealType result( zero );
      while( chunkIndex < lastChunk )
         result = reduction( result,  chunksResults[ chunkIndex++ ] );
      if( row >= first && row < last )
         keeper( row, result );
   }
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Real,
             typename... Args >
__device__
void
ChunkedEllpackView< Device, Index, Organization >::
segmentsReductionKernel( IndexType gridIdx,
                         IndexType first,
                         IndexType last,
                         Fetch fetch,
                         Reduction reduction,
                         ResultKeeper keeper,
                         Real zero,
                         Args... args ) const
{
   using RealType = decltype( fetch( IndexType(), std::declval< bool& >(), args... ) );

   const IndexType firstSlice = rowToSliceMapping[ first ];
   const IndexType lastSlice = rowToSliceMapping[ last - 1 ];

   const IndexType sliceIdx = firstSlice + gridIdx * Cuda::getMaxGridSize() + blockIdx.x;
   if( sliceIdx > lastSlice )
      return;

   RealType* chunksResults = Cuda::getSharedMemory< RealType >();
   __shared__ details::ChunkedEllpackSliceInfo< IndexType > sliceInfo;

   if( threadIdx.x == 0 )
      sliceInfo = this->slices[ sliceIdx ];
   chunksResults[ threadIdx.x ] = zero;
   __syncthreads();

   const IndexType sliceOffset = sliceInfo.pointer;
   const IndexType chunkSize = sliceInfo.chunkSize;
   //const IndexType chunkIdx = sliceIdx * chunksInSlice + threadIdx.x;
   bool compute( true );

   if( Organization == RowMajorOrder )
   {
      IndexType begin = sliceOffset + threadIdx.x * chunkSize; // threadIdx.x = chunkIdx within the slice
      IndexType end = begin + chunkSize;
      for( IndexType j = begin; j < end && compute; j++ )
         chunksResults[ threadIdx.x ] = reduction( chunksResults[ threadIdx.x ], fetch( j, compute ) );
   }
   else
   {
      const IndexType begin = sliceOffset + threadIdx.x; // threadIdx.x = chunkIdx within the slice
      const IndexType end = begin + chunksInSlice * chunkSize;
         for( IndexType j = begin; j < end && compute; j += chunksInSlice )
            chunksResults[ threadIdx.x ] = reduction( chunksResults[ threadIdx.x ], fetch( j, compute ) );
   }
   __syncthreads();

   if( threadIdx.x < sliceInfo.size )
   {
      const IndexType row = sliceInfo.firstSegment + threadIdx.x;
      IndexType chunkIndex( 0 );
      if( threadIdx.x != 0 )
         chunkIndex = this->rowToChunkMapping[ row - 1 ];
      const IndexType lastChunk = this->rowToChunkMapping[ row ];
      RealType result( zero );
      while( chunkIndex < lastChunk )
         result = reduction( result,  chunksResults[ chunkIndex++ ] );
      if( row >= first && row < last )
         keeper( row, result );
   }
}
#endif

      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL
