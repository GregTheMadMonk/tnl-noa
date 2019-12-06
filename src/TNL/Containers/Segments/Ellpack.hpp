/***************************************************************************
                          Ellpack.hpp -  description
                             -------------------
    begin                : Dec 3, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Containers/Segments/Ellpack.h>

namespace TNL {
   namespace Containers {
      namespace Segments {


template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
Ellpack< Device, Index, RowMajorOrder, Alignment >::
Ellpack()
   : segmentSize( 0 ), size( 0 ), alignedSize( 0 )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
Ellpack< Device, Index, RowMajorOrder, Alignment >::
Ellpack( const SegmentsSizes& segmentsSizes )
   : segmentSize( 0 ), size( 0 ), alignedSize( 0 )
{
   this->setSegmentsSizes( segmentsSizes );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
Ellpack< Device, Index, RowMajorOrder, Alignment >::
Ellpack( const IndexType segmentsCount, const IndexType segmentSize )
   : segmentSize( 0 ), size( 0 ), alignedSize( 0 )
{
   this->setSegmentsSizes( segmentsCount, segmentSize );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
Ellpack< Device, Index, RowMajorOrder, Alignment >::
Ellpack( const Ellpack& ellpack )
   : segmentSize( ellpack.segmentSize ), size( ellpack.size ), alignedSize( ellpack.alignedSize )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
Ellpack< Device, Index, RowMajorOrder, Alignment >::
Ellpack( const Ellpack&& ellpack )
   : segmentSize( ellpack.segmentSize ), size( ellpack.size ), alignedSize( ellpack.alignedSize )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
   template< typename SizesHolder >
void
Ellpack< Device, Index, RowMajorOrder, Alignment >::
setSegmentsSizes( const SizesHolder& sizes )
{
   this->segmentSize = max( sizes );
   this->size = sizes.getSize();
   if( RowMajorOrder )
      this->alignedSize = this->size;
   else
      this->alignedSize = roundUpDivision( size, this->getAlignment() ) * this->getAlignment();
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
void
Ellpack< Device, Index, RowMajorOrder, Alignment >::
setSegmentsSizes( const IndexType segmentsCount, const IndexType segmentSize )
{
   this->segmentSize = segmentSize;
   this->size = segmentsCount;
   if( RowMajorOrder )
      this->alignedSize = this->size;
   else
      this->alignedSize = roundUpDivision( size, this->getAlignment() ) * this->getAlignment();
}


template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
__cuda_callable__
Index
Ellpack< Device, Index, RowMajorOrder, Alignment >::
getSegmentsCount() const
{
   return this->size;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
__cuda_callable__
Index
Ellpack< Device, Index, RowMajorOrder, Alignment >::
getSegmentSize( const IndexType segmentIdx ) const
{
   return this->segmentSize;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
__cuda_callable__
Index
Ellpack< Device, Index, RowMajorOrder, Alignment >::
getSize() const
{
   return this->size * this->segmentSize;
}


template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
__cuda_callable__
Index
Ellpack< Device, Index, RowMajorOrder, Alignment >::
getStorageSize() const
{
   return this->alignedSize * this->segmentSize;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
__cuda_callable__
Index
Ellpack< Device, Index, RowMajorOrder, Alignment >::
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
          int Alignment >
__cuda_callable__
void
Ellpack< Device, Index, RowMajorOrder, Alignment >::
getSegmentAndLocalIndex( const Index globalIdx, Index& segmentIdx, Index& localIdx ) const
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
   template< typename Function, typename... Args >
void
Ellpack< Device, Index, RowMajorOrder, Alignment >::
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
          int Alignment >
   template< typename Function, typename... Args >
void
Ellpack< Device, Index, RowMajorOrder, Alignment >::
forAll( Function& f, Args... args ) const
{
   this->forSegments( 0, this->getSize(), f, args... );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
Ellpack< Device, Index, RowMajorOrder, Alignment >::
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
          int Alignment >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
Ellpack< Device, Index, RowMajorOrder, Alignment >::
allReduction( Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->segmentsReduction( 0, this->getSize(), fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
void
Ellpack< Device, Index, RowMajorOrder, Alignment >::
save( File& file ) const
{
   file.save( &segmentSize );
   file.save( &size );
   file.save( &alignedSize );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
void
Ellpack< Device, Index, RowMajorOrder, Alignment >::
load( File& file )
{
   file.load( &segmentSize );
   file.load( &size );
   file.load( &alignedSize );
}

      } // namespace Segments
   }  // namespace Conatiners
} // namespace TNL
