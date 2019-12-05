/***************************************************************************
                          SlicedEllpack.hpp -  description
                             -------------------
    begin                : Dec 4, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Containers/Segments/SlicedEllpack.h>
#include <TNL/Containers/Segments/Ellpack.h>

namespace TNL {
   namespace Containers {
      namespace Segments {


template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
SlicedEllpack< Device, Index, RowMajorOrder, SliceSize >::
SlicedEllpack()
   : size( 0 )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
SlicedEllpack< Device, Index, RowMajorOrder, SliceSize >::
SlicedEllpack( const SlicedEllpack& slicedEllpack )
   : size( slicedEllpack.size ), sliceOffsets( slicedEllpack.sliceOffsets )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
SlicedEllpack< Device, Index, RowMajorOrder, SliceSize >::
SlicedEllpack( const SlicedEllpack&& slicedEllpack )
   : size( slicedEllpack.size ), sliceOffsets( slicedEllpack.sliceOffsets )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
   template< typename SizesHolder >
void
SlicedEllpack< Device, Index, RowMajorOrder, SliceSize >::
setSizes( const SizesHolder& sizes )
{
   this->size = sizes.getSize();
   const IndexType segmentsCount = roundUpDivision( this->size, getSliceSize() );
   this->segmentOffsets.setSize( segmentsCount + 1 );
   Ellpack< DeviceType, IndexType, true > ellpack;
   ellpack.setSizes( segmentsCount, SliceSize );
   ...





   if( RowMajorOrder )
      this->alignedSize = this->size;
   else
      this->alignedSize = roundUpDivision( size, this->getSliceSize() ) * this->getSliceSize();
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
Index
SlicedEllpack< Device, Index, RowMajorOrder, SliceSize >::
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
SlicedEllpack< Device, Index, RowMajorOrder, SliceSize >::
getSegmentSize( const IndexType segmentIdx ) const
{
   return this->segmentSize;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
Index
SlicedEllpack< Device, Index, RowMajorOrder, SliceSize >::
getStorageSize() const
{
   return this->alignedSize * this->segmentSize;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
Index
SlicedEllpack< Device, Index, RowMajorOrder, SliceSize >::
getGlobalIndex( const Index segmentIdx, const Index localIdx ) const
{
   if( RowMajorOrder )
      return segmentIdx * this->segmentSize + localIdx;
   else
      return segmentIdx + this->alignedSize * localIdx;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
__cuda_callable__
void
SlicedEllpack< Device, Index, RowMajorOrder, SliceSize >::
getSegmentAndLocalIndex( const Index globalIdx, Index& segmentIdx, Index& localIdx ) const
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
   template< typename Function, typename... Args >
void
SlicedEllpack< Device, Index, RowMajorOrder, SliceSize >::
forSegments( IndexType first, IndexType last, Function& f, Args... args ) const
{
   const auto offsetsView = this->offsets.getView();
   if( RowMajorOrder )
   {
      const IndexType segmentSize = this->segmentSize;
      auto l = [=] __cuda_callable__ ( const IndexType i, Args... args ) {
         const IndexType begin = i * segmentSize;
         const IndexType end = begin + segmentSize;
         for( IndexType j = begin; j < end; j++  )
            if( ! f( i, j, args... ) )
               break;
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l, args... );
   }
   else
   {
      const IndexType storageSize = this->getStorageSize();
      const IndexType alignedSize = this->alignedSize;
      auto l = [=] __cuda_callable__ ( const IndexType i, Args... args ) {
         const IndexType begin = i;
         const IndexType end = storageSize;
         for( IndexType j = begin; j < end; j += alignedSize )
            if( ! f( i, j, args... ) )
               break;
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
SlicedEllpack< Device, Index, RowMajorOrder, SliceSize >::
forAll( Function& f, Args... args ) const
{
   this->forSegments( 0, this->getSize(), f, args... );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
SlicedEllpack< Device, Index, RowMajorOrder, SliceSize >::
segmentsReduction( IndexType first, IndexType last, Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   if( RowMajorOrder )
   {
      using RealType = decltype( fetch( IndexType(), IndexType() ) );
      const IndexType segmentSize = this->segmentSize;
      auto l = [=] __cuda_callable__ ( const IndexType i, Args... args ) mutable {
         const IndexType begin = i * segmentSize;
         const IndexType end = begin + segmentSize;
         RealType aux( zero );
         for( IndexType j = begin; j < end; j++  )
            reduction( aux, fetch( i, j, args... ) );
         keeper( i, aux );
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l, args... );
   }
   else
   {
      using RealType = decltype( fetch( IndexType(), IndexType() ) );
      const IndexType storageSize = this->getStorageSize();
      const IndexType alignedSize = this->alignedSize;
      auto l = [=] __cuda_callable__ ( const IndexType i, Args... args ) mutable {
         const IndexType begin = i;
         const IndexType end = storageSize;
         RealType aux( zero );
         for( IndexType j = begin; j < end; j += alignedSize  )
            reduction( aux, fetch( i, j, args... ) );
         keeper( i, aux );
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
SlicedEllpack< Device, Index, RowMajorOrder, SliceSize >::
allReduction( Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->segmentsReduction( 0, this->getSize(), fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
void
SlicedEllpack< Device, Index, RowMajorOrder, SliceSize >::
save( File& file ) const
{
   file.save( &segmentSize );
   file.save( &size );
   file.save( &alignedSize );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int SliceSize >
void
SlicedEllpack< Device, Index, RowMajorOrder, SliceSize >::
load( File& file )
{
   file.load( &segmentSize );
   file.load( &size );
   file.load( &alignedSize );
}

      } // namespace Segments
   }  // namespace Conatiners
} // namespace TNL
