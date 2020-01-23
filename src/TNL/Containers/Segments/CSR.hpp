/***************************************************************************
                          CSR.hpp -  description
                             -------------------
    begin                : Nov 29, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Containers/Segments/CSR.h>
#include <TNL/Containers/Segments/details/CSR.h>

namespace TNL {
   namespace Containers {
      namespace Segments {


template< typename Device,
          typename Index,
          typename IndexAllocator >
CSR< Device, Index, IndexAllocator >::
CSR()
{
}

template< typename Device,
          typename Index,
          typename IndexAllocator >
CSR< Device, Index, IndexAllocator >::
CSR( const SegmentsSizes& segmentsSizes )
{
   this->setSegmentsSizes( segmentsSizes );
}

template< typename Device,
          typename Index,
          typename IndexAllocator >
CSR< Device, Index, IndexAllocator >::
CSR( const CSR& csr ) : offsets( csr.offsets )
{
}

template< typename Device,
          typename Index,
          typename IndexAllocator >
CSR< Device, Index, IndexAllocator >::
CSR( const CSR&& csr ) : offsets( std::move( csr.offsets ) )
{

}

template< typename Device,
          typename Index,
          typename IndexAllocator >
String
CSR< Device, Index, IndexAllocator >::
getSerializationType()
{
   return "CSR< [any_device], " + TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device,
          typename Index,
          typename IndexAllocator >
   template< typename SizesHolder >
void
CSR< Device, Index, IndexAllocator >::
setSegmentsSizes( const SizesHolder& sizes )
{
   details::CSR< Device, Index >::setSegmentsSizes( sizes, this->offsets );
}

template< typename Device,
          typename Index,
          typename IndexAllocator >
typename CSR< Device, Index, IndexAllocator >::ViewType
CSR< Device, Index, IndexAllocator >::
getView()
{
   return ViewType( this->offsets.getView() );
}

template< typename Device,
          typename Index,
          typename IndexAllocator >
typename CSR< Device, Index, IndexAllocator >::ConstViewType
CSR< Device, Index, IndexAllocator >::
getConstView() const
{
   return ConstViewType( this->offsets.getConstView() );
}

template< typename Device,
          typename Index,
          typename IndexAllocator >
__cuda_callable__
Index
CSR< Device, Index, IndexAllocator >::
getSegmentsCount() const
{
   return this->offsets.getSize() - 1;
}

template< typename Device,
          typename Index,
          typename IndexAllocator >
__cuda_callable__
Index
CSR< Device, Index, IndexAllocator >::
getSegmentSize( const IndexType segmentIdx ) const
{
   return details::CSR< Device, Index >::getSegmentSize( this->offsets, segmentIdx );
}

template< typename Device,
          typename Index,
          typename IndexAllocator >
__cuda_callable__
Index
CSR< Device, Index, IndexAllocator >::
getSize() const
{
   return this->getStorageSize();
}

template< typename Device,
          typename Index,
          typename IndexAllocator >
__cuda_callable__
Index
CSR< Device, Index, IndexAllocator >::
getStorageSize() const
{
   return details::CSR< Device, Index >::getStorageSize( this->offsets );
}

template< typename Device,
          typename Index,
          typename IndexAllocator >
__cuda_callable__
Index
CSR< Device, Index, IndexAllocator >::
getGlobalIndex( const Index segmentIdx, const Index localIdx ) const
{
   if( ! std::is_same< DeviceType, Devices::Host >::value )
   {
#ifdef __CUDA_ARCH__
      return offsets[ segmentIdx ] + localIdx;
#else
      return offsets.getElement( segmentIdx ) + localIdx;
#endif
   }
   return offsets[ segmentIdx ] + localIdx;
}

template< typename Device,
          typename Index,
          typename IndexAllocator >
__cuda_callable__
void
CSR< Device, Index, IndexAllocator >::
getSegmentAndLocalIndex( const Index globalIdx, Index& segmentIdx, Index& localIdx ) const
{
}

template< typename Device,
          typename Index,
          typename IndexAllocator >
__cuda_callable__
auto
CSR< Device, Index, IndexAllocator >::
getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   return SegmentViewType( offsets[ segmentIdx ], offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ] );
}

template< typename Device,
          typename Index,
          typename IndexAllocator >
   template< typename Function, typename... Args >
void
CSR< Device, Index, IndexAllocator >::
forSegments( IndexType first, IndexType last, Function& f, Args... args ) const
{
   const auto offsetsView = this->offsets.getConstView();
   auto l = [=] __cuda_callable__ ( const IndexType segmentIdx, Args... args ) mutable {
      const IndexType begin = offsetsView[ segmentIdx ];
      const IndexType end = offsetsView[ segmentIdx + 1 ];
      IndexType localIdx( 0 );
      for( IndexType globalIdx = begin; globalIdx < end; globalIdx++  )
         if( ! f( segmentIdx, localIdx++, globalIdx, args... ) )
            break;
   };
   Algorithms::ParallelFor< Device >::exec( first, last, l, args... );
}

template< typename Device,
          typename Index,
          typename IndexAllocator>
   template< typename Function, typename... Args >
void
CSR< Device, Index, IndexAllocator >::
forAll( Function& f, Args... args ) const
{
   this->forSegments( 0, this->getSegmentsCount(), f, args... );
}

template< typename Device,
          typename Index,
          typename IndexAllocator >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
CSR< Device, Index, IndexAllocator >::
segmentsReduction( IndexType first, IndexType last, Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   using RealType = decltype( fetch( IndexType(), IndexType(), IndexType(), std::declval< bool& >(), args... ) );
   const auto offsetsView = this->offsets.getConstView();
   auto l = [=] __cuda_callable__ ( const IndexType i, Args... args ) mutable {
      const IndexType begin = offsetsView[ i ];
      const IndexType end = offsetsView[ i + 1 ];
      RealType aux( zero );
      bool compute( true );
      IndexType localIdx( 0 );
      for( IndexType j = begin; j < end && compute; j++  )
         reduction( aux, fetch( i, localIdx++, j, compute, args... ) );
      keeper( i, aux );
   };
   Algorithms::ParallelFor< Device >::exec( first, last, l, args... );
}

template< typename Device,
          typename Index,
          typename IndexAllocator >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
CSR< Device, Index, IndexAllocator >::
allReduction( Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->segmentsReduction( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index,
          typename IndexAllocator >
   template< typename Device_, typename Index_, typename IndexAllocator_ >
CSR< Device, Index, IndexAllocator >&
CSR< Device, Index, IndexAllocator >::
operator=( const CSR< Device_, Index_, IndexAllocator_ >& source )
{
   this->offsets = source.offsets;
   return *this;
}

template< typename Device,
          typename Index,
          typename IndexAllocator >
void
CSR< Device, Index, IndexAllocator >::
save( File& file ) const
{
   file << this->offsets;
}

template< typename Device,
          typename Index,
          typename IndexAllocator >
void
CSR< Device, Index, IndexAllocator >::
load( File& file )
{
   file >> this->offsets;
}

      } // namespace Segments
   }  // namespace Conatiners
} // namespace TNL
