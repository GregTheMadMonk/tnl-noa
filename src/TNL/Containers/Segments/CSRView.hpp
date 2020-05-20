/***************************************************************************
                          CSRView.hpp -  description
                             -------------------
    begin                : Dec 11, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Containers/Segments/CSRView.h>
#include <TNL/Containers/Segments/details/CSR.h>
#include <TNL/Containers/Segments/details/LambdaAdapter.h>

namespace TNL {
   namespace Containers {
      namespace Segments {


template< typename Device,
          typename Index >
__cuda_callable__
CSRView< Device, Index >::
CSRView()
{
}

template< typename Device,
          typename Index >
__cuda_callable__
CSRView< Device, Index >::
CSRView( const OffsetsView& offsets_view )
   : offsets( offsets_view )
{
}

template< typename Device,
          typename Index >
__cuda_callable__
CSRView< Device, Index >::
CSRView( const OffsetsView&& offsets_view )
   : offsets( offsets_view )
{
}

template< typename Device,
          typename Index >
__cuda_callable__
CSRView< Device, Index >::
CSRView( const CSRView& csr_view )
   : offsets( csr_view.offsets )
{
}

template< typename Device,
          typename Index >
__cuda_callable__
CSRView< Device, Index >::
CSRView( const CSRView&& csr_view )
   : offsets( std::move( csr_view.offsets ) )
{
}

template< typename Device,
          typename Index >
String
CSRView< Device, Index >::
getSerializationType()
{
   return "CSR< [any_device], " + TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device,
          typename Index >
String
CSRView< Device, Index >::
getSegmentsType()
{
   return "CSR";
}

template< typename Device,
          typename Index >
__cuda_callable__
typename CSRView< Device, Index >::ViewType
CSRView< Device, Index >::
getView()
{
   return ViewType( this->offsets );
}

template< typename Device,
          typename Index >
__cuda_callable__
auto
CSRView< Device, Index >::
getConstView() const -> const ConstViewType
{
   return ConstViewType( this->offsets.getConstView() );
}

template< typename Device,
          typename Index >
__cuda_callable__ auto CSRView< Device, Index >::
getSegmentsCount() const -> IndexType
{
   return this->offsets.getSize() - 1;
}

template< typename Device,
          typename Index >
__cuda_callable__ auto CSRView< Device, Index >::
getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   return details::CSR< Device, Index >::getSegmentSize( this->offsets, segmentIdx );
}

template< typename Device,
          typename Index >
__cuda_callable__ auto CSRView< Device, Index >::
getSize() const -> IndexType
{
   return this->getStorageSize();
}

template< typename Device,
          typename Index >
__cuda_callable__ auto CSRView< Device, Index >::
getStorageSize() const -> IndexType
{
   return details::CSR< Device, Index >::getStorageSize( this->offsets );
}

template< typename Device,
          typename Index >
__cuda_callable__ auto CSRView< Device, Index >::
getGlobalIndex( const Index segmentIdx, const Index localIdx ) const -> IndexType
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
          typename Index >
__cuda_callable__
auto
CSRView< Device, Index >::
getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   return SegmentViewType( offsets[ segmentIdx ], offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ], 1 );
}

template< typename Device,
          typename Index >
   template< typename Function, typename... Args >
void
CSRView< Device, Index >::
forSegments( IndexType first, IndexType last, Function& f, Args... args ) const
{
   const auto offsetsView = this->offsets;
   auto l = [=] __cuda_callable__ ( const IndexType segmentIdx, Args... args ) mutable {
      const IndexType begin = offsetsView[ segmentIdx ];
      const IndexType end = offsetsView[ segmentIdx + 1 ];
      IndexType localIdx( 0 );
      bool compute( true );
      for( IndexType globalIdx = begin; globalIdx < end && compute; globalIdx++  )
         f( segmentIdx, localIdx++, globalIdx, compute, args... );
   };
   Algorithms::ParallelFor< Device >::exec( first, last, l, args... );
}

template< typename Device,
          typename Index >
   template< typename Function, typename... Args >
void
CSRView< Device, Index >::
forAll( Function& f, Args... args ) const
{
   this->forSegments( 0, this->getSegmentsCount(), f, args... );
}

template< typename Device,
          typename Index >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
CSRView< Device, Index >::
segmentsReduction( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   using RealType = typename details::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   const auto offsetsView = this->offsets.getConstView();
   auto l = [=] __cuda_callable__ ( const IndexType segmentIdx, Args... args ) mutable {
      const IndexType begin = offsetsView[ segmentIdx ];
      const IndexType end = offsetsView[ segmentIdx + 1 ];
      RealType aux( zero );
      IndexType localIdx( 0 );
      bool compute( true );
      for( IndexType globalIdx = begin; globalIdx < end && compute; globalIdx++  )
         aux = reduction( aux, details::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx, compute ) );
      keeper( segmentIdx, aux );
   };
   Algorithms::ParallelFor< Device >::exec( first, last, l, args... );
}

template< typename Device,
          typename Index >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
CSRView< Device, Index >::
allReduction( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->segmentsReduction( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index >
CSRView< Device, Index >&
CSRView< Device, Index >::
operator=( const CSRView& view )
{
   this->offsets.bind( view.offsets );
   return *this;
}

template< typename Device,
          typename Index >
void
CSRView< Device, Index >::
save( File& file ) const
{
   file << this->offsets;
}

template< typename Device,
          typename Index >
void
CSRView< Device, Index >::
load( File& file )
{
   file >> this->offsets;
}

      } // namespace Segments
   }  // namespace Conatiners
} // namespace TNL
