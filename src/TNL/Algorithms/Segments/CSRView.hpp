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
#include <TNL/Algorithms/Segments/CSRView.h>
#include <TNL/Algorithms/Segments/details/CSR.h>
#include <TNL/Algorithms/Segments/details/LambdaAdapter.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {


template< typename Device,
          typename Index,
          typename Kernel >
__cuda_callable__
CSRView< Device, Index, Kernel >::
CSRView()
{
}

template< typename Device,
          typename Index,
          typename Kernel >
__cuda_callable__
CSRView< Device, Index, Kernel >::
CSRView( const OffsetsView& offsets_view,
         const KernelView& kernel_view )
   : offsets( offsets_view ), kernel( kernel_view )
{
}

template< typename Device,
          typename Index,
          typename Kernel >
__cuda_callable__
CSRView< Device, Index, Kernel >::
CSRView( const OffsetsView&& offsets_view,
         const KernelView&& kernel_view )
   : offsets( std::move( offsets_view ) ), kernel( std::move( kernel_view ) )
{
}

template< typename Device,
          typename Index,
          typename Kernel >
__cuda_callable__
CSRView< Device, Index, Kernel >::
CSRView( const CSRView& csr_view )
   : offsets( csr_view.offsets ), kernel( csr_view.kernel )
{
}

template< typename Device,
          typename Index,
          typename Kernel >
__cuda_callable__
CSRView< Device, Index, Kernel >::
CSRView( const CSRView&& csr_view )
   : offsets( std::move( csr_view.offsets ) ), kernel( std::move( csr_view.kernel ) )
{
}

template< typename Device,
          typename Index,
          typename Kernel >
String
CSRView< Device, Index, Kernel >::
getSerializationType()
{
   return "CSR< [any_device], " +
      TNL::getSerializationType< IndexType >() +
      TNL::getSerializationType< KernelType >() + " >";
}

template< typename Device,
          typename Index,
          typename Kernel >
String
CSRView< Device, Index, Kernel >::
getSegmentsType()
{
   return "CSR< " + KernelType::getKernelType() + " >";
}

template< typename Device,
          typename Index,
          typename Kernel >
__cuda_callable__
typename CSRView< Device, Index, Kernel >::ViewType
CSRView< Device, Index, Kernel >::
getView()
{
   return ViewType( this->offsets, this->kernel );
}

template< typename Device,
          typename Index,
          typename Kernel >
__cuda_callable__
auto
CSRView< Device, Index, Kernel >::
getConstView() const -> const ConstViewType
{
   return ConstViewType( this->offsets.getConstView(), this->kernel.getConstView() );
}

template< typename Device,
          typename Index,
          typename Kernel >
__cuda_callable__ auto CSRView< Device, Index, Kernel >::
getSegmentsCount() const -> IndexType
{
   return this->offsets.getSize() - 1;
}

template< typename Device,
          typename Index,
          typename Kernel >
__cuda_callable__ auto CSRView< Device, Index, Kernel >::
getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   return details::CSR< Device, Index >::getSegmentSize( this->offsets, segmentIdx );
}

template< typename Device,
          typename Index,
          typename Kernel >
__cuda_callable__ auto CSRView< Device, Index, Kernel >::
getSize() const -> IndexType
{
   return this->getStorageSize();
}

template< typename Device,
          typename Index,
          typename Kernel >
__cuda_callable__ auto CSRView< Device, Index, Kernel >::
getStorageSize() const -> IndexType
{
   return details::CSR< Device, Index >::getStorageSize( this->offsets );
}

template< typename Device,
          typename Index,
          typename Kernel >
__cuda_callable__ auto CSRView< Device, Index, Kernel >::
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
          typename Index,
          typename Kernel >
__cuda_callable__
auto
CSRView< Device, Index, Kernel >::
getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   return SegmentViewType( segmentIdx, offsets[ segmentIdx ], offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ], 1 );
}

template< typename Device,
          typename Index,
          typename Kernel >
   template< typename Function >
void
CSRView< Device, Index, Kernel >::
forElements( IndexType begin, IndexType end, Function&& f ) const
{
   const auto offsetsView = this->offsets;
   auto l = [=] __cuda_callable__ ( const IndexType segmentIdx ) mutable {
      const IndexType begin = offsetsView[ segmentIdx ];
      const IndexType end = offsetsView[ segmentIdx + 1 ];
      IndexType localIdx( 0 );
      bool compute( true );
      for( IndexType globalIdx = begin; globalIdx < end && compute; globalIdx++  )
         f( segmentIdx, localIdx++, globalIdx, compute );
   };
   Algorithms::ParallelFor< Device >::exec( begin, end, l );
}

template< typename Device,
          typename Index,
          typename Kernel >
   template< typename Function >
void
CSRView< Device, Index, Kernel >::
forAllElements( Function&& f ) const
{
   this->forElements( 0, this->getSegmentsCount(), f );
}

template< typename Device,
          typename Index,
          typename Kernel >
   template< typename Function >
void
CSRView< Device, Index, Kernel >::
forSegments( IndexType begin, IndexType end, Function&& function ) const
{
   auto view = this->getConstView();
   auto f = [=] __cuda_callable__ ( IndexType segmentIdx ) mutable {
      auto segment = view.getSegmentView( segmentIdx );
      function( segment );
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( begin, end, f );
}

template< typename Device,
          typename Index,
          typename Kernel >
   template< typename Function >
void
CSRView< Device, Index, Kernel >::
forEachSegment( Function&& f ) const
{
   this->forSegments( 0, this->getSegmentsCount(), f );
}

template< typename Device,
          typename Index,
          typename Kernel >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
CSRView< Device, Index, Kernel >::
segmentsReduction( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   if( std::is_same< DeviceType, TNL::Devices::Host >::value )
      TNL::Algorithms::Segments::CSRScalarKernel< IndexType, DeviceType >::segmentsReduction( offsets, first, last, fetch, reduction, keeper, zero, args... );
   else
      kernel.segmentsReduction( offsets, first, last, fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index,
          typename Kernel >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
CSRView< Device, Index, Kernel >::
allReduction( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->segmentsReduction( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index,
          typename Kernel >
CSRView< Device, Index, Kernel >&
CSRView< Device, Index, Kernel >::
operator=( const CSRView& view )
{
   this->offsets.bind( view.offsets );
   this->kernel = view.kernel;
   return *this;
}

template< typename Device,
          typename Index,
          typename Kernel >
void
CSRView< Device, Index, Kernel >::
save( File& file ) const
{
   file << this->offsets;
}

template< typename Device,
          typename Index,
          typename Kernel >
void
CSRView< Device, Index, Kernel >::
load( File& file )
{
   file >> this->offsets;
   this->kernel.init( this->offsets );
}

      } // namespace Segments
   }  // namespace Containers
} // namespace TNL
