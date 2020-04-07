/***************************************************************************
                          BiEllpackView.hpp -  description
                             -------------------
    begin                : Apr 5, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Containers/Segments/BiEllpackView.h>
#include <TNL/Containers/Segments/details/LambdaAdapter.h>
//#include <TNL/Containers/Segments/details/BiEllpack.h>

namespace TNL {
   namespace Containers {
      namespace Segments {

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
__cuda_callable__
BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
BiEllpackView( const IndexType size,
               const IndexType storageSize,
               const IndexType virtualRows,
               const OffsetsView& rowPermArray,
               const OffsetsView& groupPointers )
: size( size ),
  storageSize( storageSize ),
  virtualRows( virtualRows ),
  rowPermArray( rowPermArray ),
  groupPointers( groupPointers )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
__cuda_callable__
BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
BiEllpackView( const IndexType size,
               const IndexType storageSize,
               const IndexType virtualRows,
               const OffsetsView&& rowPermArray,
               const OffsetsView&& groupPointers )
: size( size ),
  storageSize( storageSize ),
  virtualRows( virtualRows ),
  rowPermArray( std::move( rowPermArray ) ),
  groupPointers( std::move( groupPointers ) )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
__cuda_callable__
BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
BiEllpackView( const BiEllpackView& bi_ellpack_view )
: size( bi_ellpack_view.size ),
  storageSize( bi_ellpack_view.storageSize ),
  virtualRows( bi_ellpack_view.virtualRows ),
  rowPermArray( bi_ellpack_view.rowPermArray ),
  groupPointers( bi_ellpack_view.groupPointers )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
__cuda_callable__
BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
BiEllpackView( const BiEllpackView&& bi_ellpack_view )
: size( bi_ellpack_view.size ),
  storageSize( bi_ellpack_view.storageSize ),
  virtualRows( bi_ellpack_view.virtualRows ),
  rowPermArray( std::move( bi_ellpack_view.rowPermArray ) ),
  groupPointers( std::move( bi_ellpack_view.groupPointers ) )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
String
BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
getSerializationType()
{
   return "BiEllpack< [any_device], " + TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
String
BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
getSegmentsType()
{
   return "BiEllpack";
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
__cuda_callable__
typename BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::ViewType
BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
getView()
{
   return ViewType( size, storageSize, virtualRows, rowPermArray.getView(), groupPointers.getView() );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
__cuda_callable__ auto BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
getConstView() const -> const ConstViewType
{
   return ConstViewType( size, storageSize, virtualRows, rowPermArray.getConstView(), groupPointers.getConstView() );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
__cuda_callable__ auto BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
getSegmentsCount() const -> IndexType
{
   return this->size;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
__cuda_callable__ auto BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   if( std::is_same< DeviceType, Devices::Host >::value )
      return details::BiEllpack< IndexType, DeviceType, RowMajorOrder, WarpSize >::getSegmentSizeDirect(
         rowPermArray,
         groupPointers,
         segmentIdx );
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef __CUDA_ARCH__
      return details::BiEllpack< IndexType, DeviceType, RowMajorOrder, WarpSize >::getSegmentSizeDirect(
         rowPermArray,
         groupPointers,
         segmentIdx );
#else
      return details::BiEllpack< IndexType, DeviceType, RowMajorOrder, WarpSize >::getSegmentSize(
         rowPermArray,
         groupPointers,
         segmentIdx );
#endif
   }
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
__cuda_callable__ auto BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
getSize() const -> IndexType
{
   return this->size;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
__cuda_callable__ auto BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
getStorageSize() const -> IndexType
{
   return this->storageSize;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
__cuda_callable__ auto BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
getGlobalIndex( const Index segmentIdx, const Index localIdx ) const -> IndexType
{
   if( std::is_same< DeviceType, Devices::Host >::value )
      return details::BiEllpack< IndexType, DeviceType, RowMajorOrder, WarpSize >::getGlobalIndexDirect(
         rowPermArray,
         groupPointers,
         segmentIdx,
         localIdx );
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef __CUDA_ARCH__
      return details::BiEllpack< IndexType, DeviceType, RowMajorOrder, WarpSize >::getGlobalIndexDirect(
         rowPermArray,
         groupPointers,
         segmentIdx,
         localIdx );
#else
      return details::BiEllpack< IndexType, DeviceType, RowMajorOrder, WarpSize >::getGlobalIndex(
         rowPermArray,
         groupPointers,
         segmentIdx,
         localIdx );
#endif
   }
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
__cuda_callable__
auto
BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   if( std::is_same< DeviceType, Devices::Host >::value )
      return details::BiEllpack< IndexType, DeviceType, RowMajorOrder, WarpSize >::getSegmentViewDirect(
         rowPermArray,
         groupPointers,
         segmentIdx );
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef __CUDA_ARCH__
      return details::BiEllpack< IndexType, DeviceType, RowMajorOrder, WarpSize >::getSegmentViewDirect(
         rowPermArray,
         groupPointers,
         segmentIdx );
#else
      return details::BiEllpack< IndexType, DeviceType, RowMajorOrder, WarpSize >::getSegmentView(
         rowPermArray,
         groupPointers,
         segmentIdx );
#endif
   }
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
   template< typename Function, typename... Args >
void
BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
forSegments( IndexType first, IndexType last, Function& f, Args... args ) const
{
   //Algorithms::ParallelFor< DeviceType >::exec( first, last , work, args... );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
   template< typename Function, typename... Args >
void
BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
forAll( Function& f, Args... args ) const
{
   this->forSegments( 0, this->getSegmentsCount(), f, args... );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
segmentsReduction( IndexType first, IndexType last, Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   using RealType = typename details::FetchLambdaAdapter< Index, Fetch >::ReturnType;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
allReduction( Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->segmentsReduction( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
BiEllpackView< Device, Index, RowMajorOrder, WarpSize >&
BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
operator=( const BiEllpackView& source )
{
   this->size = source.size;
   this->storageSize = source.storageSize;
   this->virtualRows = source.virtualRows;
   this->rowPermArray = source.rowPermArray;
   this->groupPointers = source.groupPointers;
   return *this;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
void
BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
save( File& file ) const
{
   file.save( &this->size );
   file.save( &this->storageSize );
   file.save( &this->virtualRows );
   file << this->rowPermArray
        << this->groupPointers;
}

#ifdef HAVE_CUDA
template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
   template< typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Real,
             typename... Args >
__device__
void
BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
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
   __shared__ details::BiEllpackSliceInfo< IndexType > sliceInfo;
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

   if( RowMajorOrder )
   {
      IndexType begin = sliceOffset + threadIdx.x * chunkSize; // threadIdx.x = chunkIdx within the slice
      IndexType end = begin + chunkSize;
      for( IndexType j = begin; j < end && compute; j++ )
         reduction( chunksResults[ threadIdx.x ], fetch( segmentIdx, localIdx++, j, compute ) );
   }
   else
   {
      const IndexType begin = sliceOffset + threadIdx.x; // threadIdx.x = chunkIdx within the slice
      const IndexType end = begin + chunksInSlice * chunkSize;
         for( IndexType j = begin; j < end && compute; j += chunksInSlice )
            reduction( chunksResults[ threadIdx.x ], fetch( segmentIdx, localIdx++, j, compute ) );
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
         reduction( result,  chunksResults[ chunkIndex++ ] );
      if( row >= first && row < last )
         keeper( row, result );
   }
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
   template< typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Real,
             typename... Args >
__device__
void
BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
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
   __shared__ details::BiEllpackSliceInfo< IndexType > sliceInfo;

   if( threadIdx.x == 0 )
      sliceInfo = this->slices[ sliceIdx ];
   chunksResults[ threadIdx.x ] = zero;
   __syncthreads();

   const IndexType sliceOffset = sliceInfo.pointer;
   const IndexType chunkSize = sliceInfo.chunkSize;
   const IndexType chunkIdx = sliceIdx * chunksInSlice + threadIdx.x;
   bool compute( true );

   if( RowMajorOrder )
   {
      IndexType begin = sliceOffset + threadIdx.x * chunkSize; // threadIdx.x = chunkIdx within the slice
      IndexType end = begin + chunkSize;
      for( IndexType j = begin; j < end && compute; j++ )
         reduction( chunksResults[ threadIdx.x ], fetch( j, compute ) );
   }
   else
   {
      const IndexType begin = sliceOffset + threadIdx.x; // threadIdx.x = chunkIdx within the slice
      const IndexType end = begin + chunksInSlice * chunkSize;
         for( IndexType j = begin; j < end && compute; j += chunksInSlice )
            reduction( chunksResults[ threadIdx.x ], fetch( j, compute ) );
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
         reduction( result,  chunksResults[ chunkIndex++ ] );
      if( row >= first && row < last )
         keeper( row, result );
   }
}
#endif

      } // namespace Segments
   }  // namespace Containers
} // namespace TNL
