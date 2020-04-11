/***************************************************************************
                          BiEllpack.hpp -  description
                             -------------------
    begin                : Apr 5, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <math.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Containers/Segments/BiEllpack.h>
#include <TNL/Containers/Segments/Ellpack.h>

namespace TNL {
   namespace Containers {
      namespace Segments {

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
BiEllpack( const Vector< IndexType, DeviceType, IndexType >& sizes )
{
   this->setSegmentsSizes( sizes );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
BiEllpack( const BiEllpack& biEllpack )
   : size( biEllpack.size ),
     storageSize( biEllpack.storageSize ),
     virtualRows( biEllpack.virtualRows ),
     rowPermArray( biEllpack.rowPermArray ),
     groupPointers( biEllpack.groupPointers )
{
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
BiEllpack( const BiEllpack&& biEllpack )
   : size( biEllpack.size ),
     storageSize( biEllpack.storageSize ),
     virtualRows( biEllpack.virtualRows ),
     rowPermArray( std::move( biEllpack.rowPermArray ) ),
     groupPointers( std::move( biEllpack.groupPointers ) )
{
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
String
BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
getSerializationType()
{
   return "BiEllpack< [any_device], " + TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
String
BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
getSegmentsType()
{
   return ViewType::getSegmentsType();
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
typename BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::ViewType
BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
getView()
{
   return ViewType( size, storageSize, virtualRows, rowPermArray.getView(), groupPointers.getView() );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
auto BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
getConstView() const -> const ConstViewType
{
   return ConstViewType( size, storageSize, virtualRows, rowPermArray.getConstView(), groupPointers.getConstView() );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
   template< typename SizesHolder >
void BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
performRowBubbleSort( const SizesHolder& segmentsSizes )
{
   this->rowPermArray.evaluate( [] __cuda_callable__ ( const IndexType i ) -> IndexType { return i; } );

   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      IndexType strips = this->virtualRows / getWarpSize();
      for( IndexType i = 0; i < strips; i++ )
      {
         IndexType begin = i * getWarpSize();
         IndexType end = ( i + 1 ) * getWarpSize() - 1;
         if(this->getSize() - 1 < end)
            end = this->getSize() - 1;
         bool sorted = false;
         IndexType permIndex1, permIndex2, offset = 0;
         while( !sorted )
         {
            sorted = true;
            for( IndexType j = begin + offset; j < end - offset; j++ )
            {
               for( IndexType k = begin; k < end + 1; k++ )
               {
                  if( this->rowPermArray.getElement( k ) == j )
                     permIndex1 = k;
                  if( this->rowPermArray.getElement( k ) == j + 1 )
                     permIndex2 = k;
               }
               if( segmentsSizes.getElement( permIndex1 ) < segmentsSizes.getElement( permIndex2 ) )
               {
                  IndexType temp = this->rowPermArray.getElement( permIndex1 );
                  this->rowPermArray.setElement( permIndex1, this->rowPermArray.getElement( permIndex2 ) );
                  this->rowPermArray.setElement( permIndex2, temp );
                  sorted = false;
               }
            }
            for( IndexType j = end - 1 - offset; j > begin + offset; j-- )
            {
               for( IndexType k = begin; k < end + 1; k++ )
               {
                  if( this->rowPermArray.getElement( k ) == j )
                     permIndex1 = k;
                  if( this->rowPermArray.getElement( k ) == j - 1 )
                     permIndex2 = k;
               }
               if( segmentsSizes.getElement( permIndex2 ) < segmentsSizes.getElement( permIndex1 ) )
               {
                  IndexType temp = this->rowPermArray.getElement( permIndex1 );
                  this->rowPermArray.setElement( permIndex1, this->rowPermArray.getElement( permIndex2 ) );
                  this->rowPermArray.setElement( permIndex2, temp );
                  sorted = false;
               }
            }
            offset++;
         }
      }
   }
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
   template< typename SizesHolder >
void BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
computeColumnSizes( const SizesHolder& segmentsSizes )
{
   IndexType numberOfStrips = this->virtualRows / getWarpSize();
   auto groupPointersView = this->groupPointers.getView();
   auto segmentsPermutationView = this->rowPermArray.getView();
   auto segmentsSizesView = segmentsSizes.getConstView();
   const IndexType size = this->getSize();
   Algorithms::ParallelFor< DeviceType >::exec(
      ( IndexType ) 0,
      this->virtualRows / getWarpSize(),
      [=] __cuda_callable__ ( const IndexType strip ) mutable {

         IndexType firstSegment = strip * getWarpSize();
         IndexType groupBegin = strip * ( getLogWarpSize() + 1 );
         IndexType emptyGroups = 0;

         ////
         // The last strip can be shorter
         if( strip == numberOfStrips - 1 )
         {
            IndexType segmentsCount = size - firstSegment;
            while( !( segmentsCount > TNL::pow( 2, getLogWarpSize() - 1 - emptyGroups ) ) )
               emptyGroups++;
            for( IndexType group = groupBegin; group < groupBegin + emptyGroups; group++ )
               groupPointersView[ group ] = 0;
         }

         IndexType allocatedColumns = 0;
         for( IndexType groupIdx = emptyGroups; groupIdx < getLogWarpSize(); groupIdx++ )
         {
            IndexType segmentIdx = TNL::pow( 2, getLogWarpSize() - 1 - groupIdx ) - 1;
            IndexType permSegm = 0;
            while( segmentsPermutationView[ permSegm + firstSegment ] != segmentIdx + firstSegment )
               permSegm++;
            const IndexType groupWidth = segmentsSizesView[ permSegm + firstSegment ] - allocatedColumns;
            const IndexType groupHeight = TNL::pow( 2, getLogWarpSize() - groupIdx );
            const IndexType groupSize = groupWidth * groupHeight;
            allocatedColumns = segmentsSizes[ permSegm + firstSegment ];
            groupPointersView[ groupIdx + groupBegin ] = groupSize;
         }
      } );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
   template< typename SizesHolder >
void BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
verifyRowPerm( const SizesHolder& segmentsSizes )
{
   bool ok = true;
   IndexType numberOfStrips = this->virtualRows / getWarpSize();
   for( IndexType strip = 0; strip < numberOfStrips; strip++ )
   {
      IndexType begin = strip * getWarpSize();
      IndexType end = ( strip + 1 ) * getWarpSize();
      if( this->getSize() < end )
         end = this->getSize();
      for( IndexType i = begin; i < end - 1; i++ )
      {
         IndexType permIndex1, permIndex2;
         bool first = false;
         bool second = false;
         for( IndexType j = begin; j < end; j++ )
         {
            if( this->rowPermArray.getElement( j ) == i )
            {
               permIndex1 = j;
               first = true;
            }
            if( this->rowPermArray.getElement( j ) == i + 1 )
            {
               permIndex2 = j;
               second = true;
            }
         }
         if( !first || !second )
            std::cout << "Wrong permutation!" << std::endl;
         if( segmentsSizes.getElement( permIndex1 ) >= segmentsSizes.getElement( permIndex2 ) )
            continue;
         else
            ok = false;
      }
   }
   if( !ok )
      throw( std::logic_error( "Segments permutaion verification failed." ) );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
   template< typename SizesHolder >
void BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
verifyRowLengths( const SizesHolder& segmentsSizes )
{
   bool ok = true;
   for( IndexType segmentIdx = 0; segmentIdx < this->getSize(); segmentIdx++ )
   {
      const IndexType strip = segmentIdx / getWarpSize();
      const IndexType stripLength = this->getStripLength( strip );
      const IndexType groupBegin = ( getLogWarpSize() + 1 ) * strip;
      const IndexType rowStripPerm = this->rowPermArray.getElement( segmentIdx ) - strip * getWarpSize();
      const IndexType begin = this->groupPointers.getElement( groupBegin ) * getWarpSize() + rowStripPerm * stripLength;
      IndexType elementPtr = begin;
      IndexType rowLength = 0;
      const IndexType groupsCount = details::BiEllpack< Index, Device, RowMajorOrder, WarpSize >::getActiveGroupsCount( this->rowPermArray.getConstView(), segmentIdx );
      for( IndexType group = 0; group < groupsCount; group++ )
      {
         for( IndexType i = 0; i < this->getGroupLength( strip, group ); i++ )
         {
            IndexType biElementPtr = elementPtr;
            for( IndexType j = 0; j < this->power( 2, group ); j++ )
            {
               rowLength++;
               biElementPtr += this->power( 2, getLogWarpSize() - group ) * stripLength;
            }
            elementPtr++;
         }
      }
      if( segmentsSizes.getElement( segmentIdx ) > rowLength )
         ok = false;
   }
   if( ! ok )
      throw( std::logic_error( "Segments capacities verification failed." ) );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
   template< typename SizesHolder >
void
BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
setSegmentsSizes( const SizesHolder& segmentsSizes )
{
   //if( std::is_same< DeviceType, Devices::Host >::value )
   // {
      this->size = segmentsSizes.getSize();
      if( this->size % WarpSize != 0 )
         this->virtualRows = this->size + getWarpSize() - ( this->size % getWarpSize() );
      else
         this->virtualRows = this->size;
      IndexType strips = this->virtualRows / getWarpSize();
      this->rowPermArray.setSize( this->size );
      this->groupPointers.setSize( strips * ( getLogWarpSize() + 1 ) + 1 );
      this->groupPointers = 0;

      this->performRowBubbleSort( segmentsSizes );
      this->computeColumnSizes( segmentsSizes );

      this->groupPointers.template scan< Algorithms::ScanType::Exclusive >();

      this->verifyRowPerm( segmentsSizes );
      this->verifyRowLengths( segmentsSizes );
      this->storageSize =  getWarpSize() * this->groupPointers.getElement( strips * ( getLogWarpSize() + 1 ) );
   /*}
   else
   {
      BiEllpack< Devices::Host, Index, typename Allocators::Default< Devices::Host >::template Allocator< IndexType >, RowMajorOrder > hostSegments;
      Containers::Vector< IndexType, Devices::Host, IndexType > hostSegmentsSizes( segmentsSizes );
      hostSegments.setSegmentsSizes( hostSegmentsSizes );
      *this = hostSegments;
   }*/
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
__cuda_callable__ auto BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
getSegmentsCount() const -> IndexType
{
   return this->segmentsCount;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
auto BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   return details::BiEllpack< IndexType, DeviceType, RowMajorOrder >::getSegmentSize(
      rowPermArray.getConstView(),
      groupPointers.getConstView(),
      segmentIdx );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
__cuda_callable__ auto BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
getSize() const -> IndexType
{
   return this->size;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
__cuda_callable__ auto BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
getStorageSize() const -> IndexType
{
   return this->storageSize;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
__cuda_callable__ auto BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
getGlobalIndex( const IndexType segmentIdx, const IndexType localIdx ) const -> IndexType
{
      return details::BiEllpack< IndexType, DeviceType, RowMajorOrder >::getGlobalIndex(
         rowPermArray.getConstView(),
         groupPointers.getConstView(),
         segmentIdx,
         localIdx );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
__cuda_callable__ auto BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
   template< typename Function, typename... Args >
void
BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
forSegments( IndexType first, IndexType last, Function& f, Args... args ) const
{
   this->getConstView().forSegments( first, last, f, args... );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
   template< typename Function, typename... Args >
void
BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
forAll( Function& f, Args... args ) const
{
   this->forSegments( 0, this->getSegmentsCount(), f, args... );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
segmentsReduction( IndexType first, IndexType last, Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->getConstView().segmentsReduction( first, last, fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
allReduction( Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->segmentsReduction( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
   template< typename Device_, typename Index_, typename IndexAllocator_, bool RowMajorOrder_ >
BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >&
BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
operator=( const BiEllpack< Device_, Index_, IndexAllocator_, RowMajorOrder_, WarpSize >& source )
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
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
void
BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
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
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
void
BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
load( File& file )
{
   file.load( &this->size );
   file.load( &this->storageSize );
   file.load( &this->virtualRows );
   file >> this->rowPermArray
        >> this->groupPointers;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
auto BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
getStripLength( const IndexType stripIdx ) const -> IndexType
{
   return details::BiEllpack< Index, Device, RowMajorOrder, WarpSize >::getStripLength( this->groupPointers.getConstView(), stripIdx );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          bool RowMajorOrder,
          int WarpSize >
auto BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
getGroupLength( const IndexType strip, const IndexType group ) const -> IndexType
{
   return this->groupPointers.getElement( strip * ( getLogWarpSize() + 1 ) + group + 1 )
           - this->groupPointers.getElement( strip * ( getLogWarpSize() + 1 ) + group );
}

      } // namespace Segments
   }  // namespace Conatiners
} // namespace TNL
