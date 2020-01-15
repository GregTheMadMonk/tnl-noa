/***************************************************************************
                          EllpackView.hpp -  description
                             -------------------
    begin                : Dec 12, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Containers/Segments/EllpackView.h>

namespace TNL {
   namespace Containers {
      namespace Segments {


template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
__cuda_callable__
EllpackView< Device, Index, RowMajorOrder, Alignment >::
EllpackView()
   : segmentSize( 0 ), size( 0 ), alignedSize( 0 )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
__cuda_callable__
EllpackView< Device, Index, RowMajorOrder, Alignment >::
EllpackView( IndexType segmentSize, IndexType size, IndexType alignedSize )
   : segmentSize( segmentSize ), size( size ), alignedSize( alignedSize )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
__cuda_callable__
EllpackView< Device, Index, RowMajorOrder, Alignment >::
EllpackView( const EllpackView& ellpack )
   : segmentSize( ellpack.segmentSize ), size( ellpack.size ), alignedSize( ellpack.alignedSize )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
__cuda_callable__
EllpackView< Device, Index, RowMajorOrder, Alignment >::
EllpackView( const EllpackView&& ellpack )
   : segmentSize( ellpack.segmentSize ), size( ellpack.size ), alignedSize( ellpack.alignedSize )
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
String
EllpackView< Device, Index, RowMajorOrder, Alignment >::
getSerializationType()
{
   return "Ellpack< [any_device], " + TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
typename EllpackView< Device, Index, RowMajorOrder, Alignment >::ViewType
EllpackView< Device, Index, RowMajorOrder, Alignment >::
getView()
{
   return ViewType( segmentSize, size, alignedSize );
}

/*template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
typename EllpackView< Device, Index, RowMajorOrder, Alignment >::ConstViewType
EllpackView< Device, Index, RowMajorOrder, Alignment >::
getConstView() const
{
   return ConstViewType( segmentSize, size, alignedSize );
}*/

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
__cuda_callable__
Index
EllpackView< Device, Index, RowMajorOrder, Alignment >::
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
EllpackView< Device, Index, RowMajorOrder, Alignment >::
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
EllpackView< Device, Index, RowMajorOrder, Alignment >::
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
EllpackView< Device, Index, RowMajorOrder, Alignment >::
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
EllpackView< Device, Index, RowMajorOrder, Alignment >::
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
EllpackView< Device, Index, RowMajorOrder, Alignment >::
getSegmentAndLocalIndex( const Index globalIdx, Index& segmentIdx, Index& localIdx ) const
{
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
__cuda_callable__
auto
EllpackView< Device, Index, RowMajorOrder, Alignment >::
getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   if( RowMajorOrder )
      return SegmentViewType( segmentIdx * this->segmentSize, this->segmentSize, 1 );
   else
      return SegmentViewType( segmentIdx, this->segmentSize, this->alignedSize );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
   template< typename Function, typename... Args >
void
EllpackView< Device, Index, RowMajorOrder, Alignment >::
forSegments( IndexType first, IndexType last, Function& f, Args... args ) const
{
   if( RowMajorOrder )
   {
      const IndexType segmentSize = this->segmentSize;
      auto l = [=] __cuda_callable__ ( const IndexType segmentIdx, Args... args ) mutable {
         const IndexType begin = segmentIdx * segmentSize;
         const IndexType end = begin + segmentSize;
         IndexType localIdx( 0 );
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx++  )
            if( ! f( segmentIdx, localIdx++, globalIdx,  args... ) )
               break;
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l, args... );
   }
   else
   {
      const IndexType storageSize = this->getStorageSize();
      const IndexType alignedSize = this->alignedSize;
      auto l = [=] __cuda_callable__ ( const IndexType segmentIdx, Args... args ) mutable {
         const IndexType begin = segmentIdx;
         const IndexType end = storageSize;
         IndexType localIdx( 0 );
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx += alignedSize )
            if( ! f( segmentIdx, localIdx++, globalIdx, args... ) )
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
EllpackView< Device, Index, RowMajorOrder, Alignment >::
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
EllpackView< Device, Index, RowMajorOrder, Alignment >::
segmentsReduction( IndexType first, IndexType last, Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   using RealType = decltype( fetch( IndexType(), IndexType(), IndexType(), std::declval< bool& >(), args... ) );
   if( RowMajorOrder )
   {
      const IndexType segmentSize = this->segmentSize;
      auto l = [=] __cuda_callable__ ( const IndexType i, Args... args ) mutable {
         const IndexType begin = i * segmentSize;
         const IndexType end = begin + segmentSize;
         RealType aux( zero );
         IndexType localIdx( 0 );
         bool compute( true );
         for( IndexType j = begin; j < end && compute; j++  )
            reduction( aux, fetch( i, localIdx++, j, compute, args... ) );
         keeper( i, aux );
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l, args... );
   }
   else
   {
      const IndexType storageSize = this->getStorageSize();
      const IndexType alignedSize = this->alignedSize;
      auto l = [=] __cuda_callable__ ( const IndexType i, Args... args ) mutable {
         const IndexType begin = i;
         const IndexType end = storageSize;
         RealType aux( zero );
         IndexType localIdx( 0 );
         bool compute( true );
         for( IndexType j = begin; j < end && compute; j += alignedSize  )
            reduction( aux, fetch( i, localIdx++, j, compute, args... ) );
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
EllpackView< Device, Index, RowMajorOrder, Alignment >::
allReduction( Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->segmentsReduction( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
EllpackView< Device, Index, RowMajorOrder, Alignment >&
EllpackView< Device, Index, RowMajorOrder, Alignment >::
operator=( const EllpackView< Device, Index, RowMajorOrder, Alignment >& view )
{
   this->segmentSize = view.segmentSize;
   this->size = view.size;
   this->alignedSize = view.alignedSize;
}

template< typename Device,
          typename Index,
          bool RowMajorOrder,
          int Alignment >
void
EllpackView< Device, Index, RowMajorOrder, Alignment >::
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
EllpackView< Device, Index, RowMajorOrder, Alignment >::
load( File& file )
{
   file.load( &segmentSize );
   file.load( &size );
   file.load( &alignedSize );
}

      } // namespace Segments
   }  // namespace Conatiners
} // namespace TNL
