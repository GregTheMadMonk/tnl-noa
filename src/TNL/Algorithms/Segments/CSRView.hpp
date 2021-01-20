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
          CSRKernelTypes KernelType_ >
__cuda_callable__
CSRView< Device, Index, KernelType_ >::
CSRView()
{
}

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ >
__cuda_callable__
CSRView< Device, Index, KernelType_ >::
CSRView( const OffsetsView& offsets_view )
   : offsets( offsets_view )
{
}

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ >
__cuda_callable__
CSRView< Device, Index, KernelType_ >::
CSRView( const OffsetsView&& offsets_view )
   : offsets( offsets_view )
{
}

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ >
__cuda_callable__
CSRView< Device, Index, KernelType_ >::
CSRView( const CSRView& csr_view )
   : offsets( csr_view.offsets )
{
}

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ >
__cuda_callable__
CSRView< Device, Index, KernelType_ >::
CSRView( const CSRView&& csr_view )
   : offsets( std::move( csr_view.offsets ) )
{
}

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ >
String
CSRView< Device, Index, KernelType_ >::
getSerializationType()
{
   return "CSR< [any_device], " + TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ >
String
CSRView< Device, Index, KernelType_ >::
getSegmentsType()
{
   return "CSR";
}

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ >
__cuda_callable__
typename CSRView< Device, Index, KernelType_ >::ViewType
CSRView< Device, Index, KernelType_ >::
getView()
{
   return ViewType( this->offsets );
}

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ >
__cuda_callable__
auto
CSRView< Device, Index, KernelType_ >::
getConstView() const -> const ConstViewType
{
   return ConstViewType( this->offsets.getConstView() );
}

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ >
__cuda_callable__ auto CSRView< Device, Index, KernelType_ >::
getSegmentsCount() const -> IndexType
{
   return this->offsets.getSize() - 1;
}

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ >
__cuda_callable__ auto CSRView< Device, Index, KernelType_ >::
getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   return details::CSR< Device, Index >::getSegmentSize( this->offsets, segmentIdx );
}

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ >
__cuda_callable__ auto CSRView< Device, Index, KernelType_ >::
getSize() const -> IndexType
{
   return this->getStorageSize();
}

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ >
__cuda_callable__ auto CSRView< Device, Index, KernelType_ >::
getStorageSize() const -> IndexType
{
   return details::CSR< Device, Index >::getStorageSize( this->offsets );
}

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ >
__cuda_callable__ auto CSRView< Device, Index, KernelType_ >::
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
          CSRKernelTypes KernelType_ >
__cuda_callable__
auto
CSRView< Device, Index, KernelType_ >::
getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   return SegmentViewType( offsets[ segmentIdx ], offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ], 1 );
}

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ >
   template< typename Function, typename... Args >
void
CSRView< Device, Index, KernelType_ >::
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
          typename Index,
          CSRKernelTypes KernelType_ >
   template< typename Function, typename... Args >
void
CSRView< Device, Index, KernelType_ >::
forAll( Function& f, Args... args ) const
{
   this->forSegments( 0, this->getSegmentsCount(), f, args... );
}

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
CSRView< Device, Index, KernelType_ >::
segmentsReduction( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   using RealType = typename details::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   const auto offsetsView = this->offsets.getConstView();
   if( KernelType == CSRScalarKernel )
   {
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
}

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
CSRView< Device, Index, KernelType_ >::
allReduction( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->segmentsReduction( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ >
CSRView< Device, Index, KernelType_ >&
CSRView< Device, Index, KernelType_ >::
operator=( const CSRView& view )
{
   this->offsets.bind( view.offsets );
   return *this;
}

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ >
void
CSRView< Device, Index, KernelType_ >::
save( File& file ) const
{
   file << this->offsets;
}

template< typename Device,
          typename Index,
          CSRKernelTypes KernelType_ >
void
CSRView< Device, Index, KernelType_ >::
load( File& file )
{
   file >> this->offsets;
}

      } // namespace Segments
   }  // namespace Containers
} // namespace TNL
