/***************************************************************************
                          BiEllpack.hpp -  description
                             -------------------
    begin                : Apr 5, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

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
void
BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
setSegmentsSizes( const SizesHolder& segmentsSizes )
{
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
   }
   else
   {
      BiEllpack< Devices::Host, Index, typename Allocators::Default< Devices::Host >::template Allocator< Index >, RowMajorOrder > hostSegments;
      Containers::Vector< IndexType, Devices::Host, IndexType > hostSegmentsSizes( segmentsSizes );
      hostSegments.setSegmentsSizes( hostSegmentsSizes );
      *this = hostSegments;
   }
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
getGlobalIndex( const Index segmentIdx, const Index localIdx ) const -> IndexType
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
void
BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >::
printStructure( std::ostream& str )
{
   this->getView().printStructure( str );
}

      } // namespace Segments
   }  // namespace Conatiners
} // namespace TNL
