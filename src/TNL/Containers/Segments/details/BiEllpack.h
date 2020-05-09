/***************************************************************************
                          BiEllpack.h -  description
                             -------------------
    begin                : Apr 7, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Segments/BiEllpackSegmentView.h>
#include <TNL/Containers/Segments/details/CheckLambdas.h>

namespace TNL {
   namespace Containers {
      namespace Segments {
         namespace details {

template< typename Index,
          typename Device,
          bool RowMajorOrder = std::is_same< Device, Devices::Host >::value,
          int WarpSize = 32 >
class BiEllpack
{
   public:

      using DeviceType = Device;
      using IndexType = Index;
      static constexpr bool getRowMajorOrder() { return RowMajorOrder; }
      using OffsetsHolder = Containers::Vector< IndexType, DeviceType, IndexType >;
      using OffsetsHolderView = typename OffsetsHolder::ViewType;
      using ConstOffsetsHolderView = typename OffsetsHolderView::ConstViewType;
      using SegmentsSizes = OffsetsHolder;
      using SegmentViewType = BiEllpackSegmentView< IndexType, RowMajorOrder >;
      
      static constexpr int getWarpSize() { return WarpSize; };

      static constexpr int getLogWarpSize() { return std::log2( WarpSize ); };

      static constexpr int getGroupsCount() { return getLogWarpSize() + 1; };

      __cuda_callable__
      static IndexType getActiveGroupsCountDirect( const ConstOffsetsHolderView& rowPermArray, const IndexType segmentIdx )
      {
         TNL_ASSERT_GE( segmentIdx, 0, "" );
         //TNL_ASSERT_LT( segmentIdx, this->getSize(), "" );

         IndexType strip = segmentIdx / getWarpSize();
         IndexType rowStripPermutation = rowPermArray[ segmentIdx ] - getWarpSize() * strip;
         IndexType numberOfGroups = getLogWarpSize() + 1;
         IndexType bisection = 1;
         for( IndexType i = 0; i < getLogWarpSize() + 1; i++ )
         {
            if( rowStripPermutation < bisection )
               return numberOfGroups - i;
            bisection *= 2;
         }
      }

      static IndexType getActiveGroupsCount( const ConstOffsetsHolderView& rowPermArray, const IndexType segmentIdx )
      {
         TNL_ASSERT_GE( segmentIdx, 0, "" );
         //TNL_ASSERT_LT( segmentIdx, this->getSize(), "" );

         IndexType strip = segmentIdx / getWarpSize();
         IndexType rowStripPermutation = rowPermArray.getElement( segmentIdx ) - getWarpSize() * strip;
         IndexType numberOfGroups = getLogWarpSize() + 1;
         IndexType bisection = 1;
         for( IndexType i = 0; i < getLogWarpSize() + 1; i++ )
         {
            if( rowStripPermutation < bisection )
               return numberOfGroups - i;
            bisection *= 2;
         }
         throw std::logic_error( "segmentIdx was not found" );
      }

      __cuda_callable__
      static IndexType getGroupSizeDirect( const ConstOffsetsHolderView& groupPointers,
                                           const IndexType strip,
                                           const IndexType group )
      {
         const IndexType groupOffset = strip * ( getLogWarpSize() + 1 ) + group;
         return groupPointers[ groupOffset + 1 ] - groupPointers[ groupOffset ];
      }

      
      static IndexType getGroupSize( const ConstOffsetsHolderView& groupPointers,
                                     const IndexType strip,
                                     const IndexType group )
      {
         const IndexType groupOffset = strip * ( getLogWarpSize() + 1 ) + group;
         return groupPointers.getElement( groupOffset + 1 ) - groupPointers.getElement( groupOffset );
      }
      __cuda_callable__ static
      IndexType getSegmentSizeDirect( const OffsetsHolderView& rowPermArray,
                                      const OffsetsHolderView& groupPointers,
                                      const IndexType segmentIdx )
      {
         const IndexType strip = segmentIdx / getWarpSize();
         const IndexType groupIdx = strip * ( getLogWarpSize() + 1 );
         const IndexType rowStripPerm = rowPermArray[ segmentIdx ] - strip * getWarpSize();
         const IndexType groupsCount = getActiveGroupsCountDirect( rowPermArray, segmentIdx );
         IndexType groupHeight = getWarpSize();
         IndexType segmentSize = 0;
         for( IndexType groupIdx = 0; groupIdx < groupsCount; groupIdx++ )
         {
            const IndexType groupSize = getGroupSizeDirect( groupPointers, strip, groupIdx );
            IndexType groupWidth =  groupSize / groupHeight;
            segmentSize += groupWidth;
            groupHeight /= 2;
         }
         return segmentSize;
      }

      static
      IndexType getSegmentSize( const OffsetsHolderView& rowPermArray,
                                const OffsetsHolderView& groupPointers,
                                const IndexType segmentIdx )
      {
         const IndexType strip = segmentIdx / getWarpSize();
         const IndexType groupIdx = strip * ( getLogWarpSize() + 1 );
         const IndexType rowStripPerm = rowPermArray.getElement( segmentIdx ) - strip * getWarpSize();
         const IndexType groupsCount = getActiveGroupsCount( rowPermArray, segmentIdx );
         IndexType groupHeight = getWarpSize();
         IndexType segmentSize = 0;
         for( IndexType group = 0; group < groupsCount; group++ )
         {
            const IndexType groupSize = getGroupSize( groupPointers, strip, group );
            IndexType groupWidth =  groupSize / groupHeight;
            segmentSize += groupWidth;
            groupHeight /= 2;
         }
         return segmentSize;
      }

      __cuda_callable__ static
      IndexType getGlobalIndexDirect( const OffsetsHolderView& rowPermArray,
                                      const OffsetsHolderView& groupPointers,
                                      const IndexType segmentIdx,
                                      IndexType localIdx )
      {
         const IndexType strip = segmentIdx / getWarpSize();
         const IndexType groupIdx = strip * ( getLogWarpSize() + 1 );
         const IndexType rowStripPerm = rowPermArray[ segmentIdx ] - strip * getWarpSize();
         const IndexType groupsCount = getActiveGroupsCountDirect( rowPermArray, segmentIdx );
         IndexType globalIdx = groupPointers[ groupIdx ];
         IndexType groupHeight = getWarpSize();
         for( IndexType group = 0; group < groupsCount; group++ )
         {
            const IndexType groupSize = getGroupSizeDirect( groupPointers, strip, group );
            if(  groupSize )
            {
               IndexType groupWidth =  groupSize / groupHeight;
               if( localIdx >= groupWidth )
               {
                  localIdx -= groupWidth;
                  globalIdx += groupSize;
               }
               else
               {
                  if( RowMajorOrder )
                     return globalIdx + rowStripPerm * groupWidth + localIdx;
                  else
                     return globalIdx + rowStripPerm + localIdx * groupHeight;
               }
            }
            groupHeight /= 2;
         }
         TNL_ASSERT_TRUE( false, "Segment capacity exceeded, wrong localIdx." );
         return -1; // to avoid compiler warning
      }

      static
      IndexType getGlobalIndex( const ConstOffsetsHolderView& rowPermArray,
                                const ConstOffsetsHolderView& groupPointers,
                                const IndexType segmentIdx,
                                IndexType localIdx )
      {
         const IndexType strip = segmentIdx / getWarpSize();
         const IndexType groupIdx = strip * ( getLogWarpSize() + 1 );
         const IndexType rowStripPerm = rowPermArray.getElement( segmentIdx ) - strip * getWarpSize();
         const IndexType groupsCount = getActiveGroupsCount( rowPermArray, segmentIdx );
         IndexType globalIdx = groupPointers.getElement( groupIdx );
         IndexType groupHeight = getWarpSize();
         for( IndexType group = 0; group < groupsCount; group++ )
         {
            const IndexType groupSize = getGroupSize( groupPointers, strip, group );
            if(  groupSize )
            {
               IndexType groupWidth =  groupSize / groupHeight;
               if( localIdx >= groupWidth )
               {
                  localIdx -= groupWidth;
                  globalIdx += groupSize;
               }
               else
               {
                  if( RowMajorOrder )
                  {
                     return globalIdx + rowStripPerm * groupWidth + localIdx;
                  }
                  else
                     return globalIdx + rowStripPerm + localIdx * groupHeight;
               }
            }
            groupHeight /= 2;
         }
         TNL_ASSERT_TRUE( false, "Segment capacity exceeded, wrong localIdx." );
      }

      static __cuda_callable__
      SegmentViewType getSegmentViewDirect( const OffsetsHolderView& rowPermArray,
                                            const OffsetsHolderView& groupPointers,
                                            const IndexType segmentIdx )
      {
         using GroupsWidthType = typename SegmentViewType::GroupsWidthType;

         const IndexType strip = segmentIdx / getWarpSize();
         const IndexType groupIdx = strip * ( getLogWarpSize() + 1 );
         const IndexType inStripIdx = rowPermArray[ segmentIdx ] - strip * getWarpSize();
         const IndexType groupsCount = getActiveGroupsCountDirect( rowPermArray, segmentIdx );
         IndexType groupHeight = getWarpSize();
         GroupsWidthType groupsWidth( 0 );
         TNL_ASSERT_LE( groupsCount, getGroupsCount(), "" );
         for( IndexType i = 0; i < groupsCount; i++ )
         {
            const IndexType groupSize = groupPointers[ groupIdx + i + 1 ] - groupPointers[ groupIdx + i ];
            groupsWidth[ i ] = groupSize / groupHeight;
            groupHeight /= 2;
            //std::cerr << " ROW INIT: groupIdx = " << i << " groupSize = " << groupSize << " groupWidth = " << groupsWidth[ i ] << std::endl;
         }
         return SegmentViewType( groupPointers[ groupIdx ],
                                 inStripIdx,
                                 groupsWidth );
      }

      static __cuda_callable__
      SegmentViewType getSegmentView( const OffsetsHolderView& rowPermArray,
                                      const OffsetsHolderView& groupPointers,
                                      const IndexType segmentIdx )
      {
         using GroupsWidthType = typename SegmentViewType::GroupsWidthType;

         const IndexType strip = segmentIdx / getWarpSize();
         const IndexType groupIdx = strip * ( getLogWarpSize() + 1 );
         const IndexType inStripIdx = rowPermArray.getElement( segmentIdx ) - strip * getWarpSize();
         const IndexType groupsCount = getActiveGroupsCount( rowPermArray, segmentIdx );
         IndexType groupHeight = getWarpSize();
         GroupsWidthType groupsWidth( 0 );
         for( IndexType i = 0; i < groupsCount; i++ )
         {
            const IndexType groupSize = groupPointers.getElement( groupIdx + i + 1 ) - groupPointers.getElement( groupIdx + i );
            groupsWidth[ i ] = groupSize / groupHeight;
            groupHeight /= 2;
         }
         return SegmentViewType( groupPointers[ groupIdx ],
                                 inStripIdx,
                                 groupsWidth );
      }

      static
      Index getStripLength( const ConstOffsetsHolderView& groupPointers, const IndexType strip )
      {
         TNL_ASSERT( strip >= 0, std::cerr << "strip = " << strip );

          return groupPointers.getElement( ( strip + 1 ) * ( getLogWarpSize() + 1 ) )
                 - groupPointers.getElement( strip * ( getLogWarpSize() + 1 ) );
      }

      static __cuda_callable__
      Index getStripLengthDirect( const ConstOffsetsHolderView& groupPointers, const IndexType strip )
      {
         TNL_ASSERT( strip >= 0, std::cerr << "strip = " << strip );

          return groupPointers[ ( strip + 1 ) * ( getLogWarpSize() + 1 ) ]
                 - groupPointers[ strip * ( getLogWarpSize() + 1 ) ];
      }

};

#ifdef HAVE_CUDA
template< typename Index,
          typename Fetch,
          int BlockDim = 256,
          int WarpSize = 32,
          bool HasAllParameters = details::CheckFetchLambda< Index, Fetch >::hasAllParameters() >
struct BiEllpackSegmentsReductionDispatcher{};

template< typename Index, typename Fetch, int BlockDim, int WarpSize >
struct BiEllpackSegmentsReductionDispatcher< Index, Fetch, BlockDim, WarpSize, true >
{
   template< typename View,
             typename Reduction,
             typename ResultKeeper,
             typename Real,
             typename... Args >
   __device__
   static void exec( View biEllpack,
                     Index gridIdx,
                     Index first,
                     Index last,
                     Fetch fetch,
                     Reduction reduction,
                     ResultKeeper keeper,
                     Real zero,
                     Args... args )
   {
      biEllpack.template segmentsReductionKernelWithAllParameters< Fetch, Reduction, ResultKeeper, Real, BlockDim, Args... >( gridIdx, first, last, fetch, reduction, keeper, zero, args... );
   }
};

template< typename Index, typename Fetch, int BlockDim, int WarpSize >
struct BiEllpackSegmentsReductionDispatcher< Index, Fetch, BlockDim, WarpSize, false >
{
   template< typename View,
             typename Reduction,
             typename ResultKeeper,
             typename Real,
             typename... Args >
   __device__
   static void exec( View biEllpack,
                     Index gridIdx,
                     Index first,
                     Index last,
                     Fetch fetch,
                     Reduction reduction,
                     ResultKeeper keeper,
                     Real zero,
                     Args... args )
   {
      biEllpack.template segmentsReductionKernel< Fetch, Reduction, ResultKeeper, Real, BlockDim, Args... >( gridIdx, first, last, fetch, reduction, keeper, zero, args... );
   }
};

template< typename View,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Real,
          int BlockDim,
          typename... Args >
__global__
void BiEllpackSegmentsReductionKernel( View biEllpack,
                                       Index gridIdx,
                                       Index first,
                                       Index last,
                                       Fetch fetch,
                                       Reduction reduction,
                                       ResultKeeper keeper,
                                       Real zero,
                                       Args... args )
{
   BiEllpackSegmentsReductionDispatcher< Index, Fetch, BlockDim >::exec( biEllpack, gridIdx, first, last, fetch, reduction, keeper, zero, args... );
}
#endif

         } //namespace details
      } //namespace Segments
   } //namespace Containers
} //namepsace TNL
