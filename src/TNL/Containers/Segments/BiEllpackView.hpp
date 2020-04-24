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
   const auto segmentsPermutationView = this->rowPermArray.getConstView();
   const auto groupPointersView = this->groupPointers.getConstView();
   auto work = [=] __cuda_callable__ ( IndexType segmentIdx, Args... args ) mutable {
      const IndexType strip = segmentIdx / getWarpSize();
      const IndexType firstGroupInStrip = strip * ( getLogWarpSize() + 1 );
      const IndexType rowStripPerm = segmentsPermutationView[ segmentIdx ] - strip * getWarpSize();
      const IndexType groupsCount = details::BiEllpack< IndexType, DeviceType, RowMajorOrder, getWarpSize() >::getActiveGroupsCountDirect( segmentsPermutationView, segmentIdx );
      IndexType groupHeight = getWarpSize();
      //printf( "segmentIdx = %d strip = %d firstGroupInStrip = %d rowStripPerm = %d groupsCount = %d \n", segmentIdx, strip, firstGroupInStrip, rowStripPerm, groupsCount );
      bool compute( true );
      IndexType localIdx( 0 );
      for( IndexType groupIdx = firstGroupInStrip; groupIdx < firstGroupInStrip + groupsCount && compute; groupIdx++ )
      {
         IndexType groupOffset = groupPointersView[ groupIdx ];
         const IndexType groupSize = groupPointersView[ groupIdx + 1 ] - groupOffset;
         //printf( "groupSize = %d \n", groupSize );
         if( groupSize )
         {
            const IndexType groupWidth = groupSize / groupHeight;
            for( IndexType i = 0; i < groupWidth; i++ )
            {
               if( RowMajorOrder )
               {
                  f( segmentIdx, localIdx, groupOffset + rowStripPerm * groupWidth + i, compute );
               }
               else
               {
                  /*printf( "segmentIdx = %d localIdx = %d globalIdx = %d groupIdx = %d groupSize = %d groupWidth = %d\n",
                     segmentIdx, localIdx, groupOffset + rowStripPerm + i * groupHeight,
                     groupIdx, groupSize, groupWidth );*/
                  f( segmentIdx, localIdx, groupOffset + rowStripPerm + i * groupHeight, compute );
               }
               localIdx++;
            }
         }
         groupHeight /= 2;
      }
   };
   Algorithms::ParallelFor< DeviceType >::exec( first, last , work, args... );
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
segmentsReduction( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   using RealType = typename details::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   if( std::is_same< DeviceType, Devices::Host >::value )
      for( IndexType segmentIdx = 0; segmentIdx < this->getSize(); segmentIdx++ )
      {
         const IndexType stripIdx = segmentIdx / getWarpSize();
         const IndexType groupIdx = stripIdx * ( getLogWarpSize() + 1 );
         const IndexType inStripIdx = rowPermArray[ segmentIdx ] - stripIdx * getWarpSize();
         const IndexType groupsCount = details::BiEllpack< IndexType, DeviceType, RowMajorOrder, getWarpSize() >::getActiveGroupsCount( rowPermArray, segmentIdx );
         IndexType globalIdx = groupPointers[ groupIdx ];
         IndexType groupHeight = getWarpSize();
         IndexType localIdx( 0 );
         RealType aux( zero );
         bool compute( true );
         for( IndexType group = 0; group < groupsCount && compute; group++ )
         {
            const IndexType groupSize = details::BiEllpack< IndexType, DeviceType, RowMajorOrder, getWarpSize() >::getGroupSize( groupPointers, stripIdx, group );
            IndexType groupWidth = groupSize / groupHeight;
            const IndexType globalIdxBack = globalIdx;
            if( RowMajorOrder )
               globalIdx += inStripIdx * groupWidth;
            else
               globalIdx += inStripIdx;
            for( IndexType j = 0; j < groupWidth && compute; j++ )
            {
               //std::cerr << "segmentIdx = " << segmentIdx << " groupIdx = " << groupIdx 
               //         << " groupWidth = " << groupWidth << " groupHeight = " << groupHeight
               //          << " localIdx = " << localIdx << " globalIdx = " << globalIdx 
               //          << " fetch = " << details::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx, compute ) << std::endl;
               aux = reduction( aux, details::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx, compute ) );
               if( RowMajorOrder )
                  globalIdx ++;
               else
                  globalIdx += groupHeight;
            }
            globalIdx = globalIdxBack + groupSize;
            groupHeight /= 2;
         }
         keeper( segmentIdx, aux );
      }
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      constexpr int BlockDim = 256;//getWarpSize();
      dim3 cudaBlockSize = BlockDim;
      const IndexType stripsCount = roundUpDivision( last - first, getWarpSize() );
      const IndexType cudaBlocks = roundUpDivision( stripsCount * getWarpSize(), cudaBlockSize.x );
      const IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
      IndexType sharedMemory = 0;
      if( ! RowMajorOrder )
         sharedMemory = cudaBlockSize.x * sizeof( RealType );

      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
      {
         dim3 cudaGridSize = Cuda::getMaxGridSize();
         if( gridIdx == cudaGrids - 1 )
            cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
         details::BiEllpackSegmentsReductionKernel< ViewType, IndexType, Fetch, Reduction, ResultKeeper, Real, BlockDim, Args...  >
            <<< cudaGridSize, cudaBlockSize, sharedMemory  >>>
            ( *this, gridIdx, first, last, fetch, reduction, keeper, zero, args... );
         cudaThreadSynchronize();
         TNL_CHECK_CUDA_DEVICE;
      }
#endif
   }
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
allReduction( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
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
   this->rowPermArray.bind( source.rowPermArray );
   this->groupPointers.bind( source.groupPointers );
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

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
void
BiEllpackView< Device, Index, RowMajorOrder, WarpSize >::
printStructure( std::ostream& str ) const
{
   const IndexType stripsCount = roundUpDivision( this->getSize(), getWarpSize() );
   for( IndexType stripIdx = 0; stripIdx < stripsCount; stripIdx++ )
   {
      str << "Strip: " << stripIdx << std::endl;
      const IndexType firstGroupIdx = stripIdx * ( getLogWarpSize() + 1 );
      const IndexType lastGroupIdx = firstGroupIdx + getLogWarpSize() + 1;
      IndexType groupHeight = getWarpSize();
      for( IndexType groupIdx = firstGroupIdx; groupIdx < lastGroupIdx; groupIdx ++ )
      {
         const IndexType groupSize = groupPointers.getElement( groupIdx + 1 ) - groupPointers.getElement( groupIdx );
         const IndexType groupWidth = groupSize / groupHeight;
         str << "\tGroup: " << groupIdx << " size = " << groupSize << " width = " << groupWidth << " height = " << groupHeight << std::endl;
         groupHeight /= 2;
      }
   }
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
             int BlockDim,
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
   const IndexType segmentIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x + first;
   if( segmentIdx >= last )
      return;

   const IndexType strip = segmentIdx / getWarpSize();
   const IndexType firstGroupInStrip = strip * ( getLogWarpSize() + 1 );
   const IndexType rowStripPerm = rowPermArray[ segmentIdx ] - strip * getWarpSize();
   const IndexType groupsCount = details::BiEllpack< IndexType, DeviceType, RowMajorOrder, getWarpSize() >::getActiveGroupsCountDirect( rowPermArray, segmentIdx );
   IndexType groupHeight = getWarpSize();
   bool compute( true );
   IndexType localIdx( 0 );
   RealType result( zero );
   for( IndexType groupIdx = firstGroupInStrip; groupIdx < firstGroupInStrip + groupsCount && compute; groupIdx++ )
   {
      IndexType groupOffset = groupPointers[ groupIdx ];
      const IndexType groupSize = groupPointers[ groupIdx + 1 ] - groupOffset;
      if( groupSize )
      {
         const IndexType groupWidth = groupSize / groupHeight;
         for( IndexType i = 0; i < groupWidth; i++ )
         {
            if( RowMajorOrder )
               result = reduction( result, fetch( segmentIdx, localIdx, groupOffset + rowStripPerm * groupWidth + i, compute ) );
            else
               result = reduction( result, fetch( segmentIdx, localIdx, groupOffset + rowStripPerm + i * groupHeight, compute ) );
            localIdx++;
         }
      }
      groupHeight /= 2;
   }
   keeper( segmentIdx, result );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int WarpSize >
   template< typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Real,
             int BlockDim,
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
   Index segmentIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x + first;

   const IndexType strip = segmentIdx >> getLogWarpSize();
   const IndexType warpStart = strip << getLogWarpSize();
   const IndexType inWarpIdx = segmentIdx & ( getWarpSize() - 1 );

   if( warpStart >= last )
      return;

   IndexType groupHeight = getWarpSize();
   IndexType firstGroupIdx = strip * ( getLogWarpSize() + 1 );

   __shared__ RealType results[ BlockDim ];
   results[ threadIdx.x ] = zero;
   __shared__ IndexType sharedGroupPointers[ 7 ]; // TODO: getLogWarpSize() + 1 ];

   if( threadIdx.x <= getLogWarpSize() + 1 )
      sharedGroupPointers[ threadIdx.x ] = this->groupPointers[ firstGroupIdx + threadIdx.x ];
   __syncthreads();

   bool compute( true );
   if( RowMajorOrder )
   {
      for( IndexType group = 0; group < getLogWarpSize() + 1; group++ )
      {
         IndexType groupBegin = sharedGroupPointers[ group ];
         IndexType groupEnd = sharedGroupPointers[ group + 1 ];
         if( groupEnd - groupBegin > 0 )
         {

               if( inWarpIdx < groupHeight )
               {
                  const IndexType groupWidth = ( groupEnd - groupBegin ) / groupHeight;
                  IndexType globalIdx = groupBegin + inWarpIdx * groupWidth;
                  for( IndexType i = 0; i < groupWidth && compute; i++ )
                     results[ threadIdx.x ] = reduction( results[ threadIdx.x ], fetch( globalIdx++, compute ) );
               }
            }
         groupHeight >>= 1;
      }
   }
   else
   {
      RealType* temp = Cuda::getSharedMemory< RealType >();
      for( IndexType group = 0; group < getLogWarpSize() + 1; group++ )
      {
         IndexType groupBegin = sharedGroupPointers[ group ];
         IndexType groupEnd = sharedGroupPointers[ group + 1 ];
         if( groupEnd - groupBegin > 0 )
         {
            temp[ threadIdx.x ] = zero;
            IndexType globalIdx = groupBegin + inWarpIdx;
            while( globalIdx < groupEnd )
            {
               temp[ threadIdx.x ] = reduction( temp[ threadIdx.x ], fetch( globalIdx, compute ) );
               globalIdx += getWarpSize();
            }
            // TODO: reduction via templates
            IndexType bisection2 = getWarpSize();
            for( IndexType i = 0; i < group; i++ )
            {
               bisection2 >>= 1;
               if( inWarpIdx < bisection2 )
                  temp[ threadIdx.x ] = reduction( temp[ threadIdx.x ], temp[ threadIdx.x + bisection2 ] );
            }
            if( inWarpIdx < groupHeight )
               results[ threadIdx.x ] = reduction( results[ threadIdx.x ], temp[ threadIdx.x ] );
         }
         groupHeight >>= 1;
      }
   }
   __syncthreads();
   if( warpStart + inWarpIdx >= last )
      return;

   keeper( warpStart + inWarpIdx, results[ this->rowPermArray[ warpStart + inWarpIdx ] & ( blockDim.x - 1 ) ] );
}
#endif

      } // namespace Segments
   }  // namespace Containers
} // namespace TNL
