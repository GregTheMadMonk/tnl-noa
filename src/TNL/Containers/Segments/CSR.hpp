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

namespace TNL {
   namespace Containers {
      namespace Segments {


template< typename Device,
          typename Index >
CSR< Device, Index >::
CSR()
{
}

template< typename Device,
          typename Index >
CSR< Device, Index >::
CSR( const SegmentsSizes& segmentsSizes )
{
   this->setSegmentsSizes( segmentsSizes );
}

template< typename Device,
          typename Index >
CSR< Device, Index >::
CSR( const CSR& csr ) : offsets( csr.offsets )
{
}

template< typename Device,
          typename Index >
CSR< Device, Index >::
CSR( const CSR&& csr ) : offsets( std::move( csr.offsets ) )
{

}

template< typename Device,
          typename Index >
   template< typename SizesHolder >
void
CSR< Device, Index >::
setSegmentsSizes( const SizesHolder& sizes )
{
   this->offsets.setSize( sizes.getSize() + 1 );
   auto view = this->offsets.getView( 0, sizes.getSize() );
   view = sizes;
   this->offsets.setElement( sizes.getSize(), 0 );
   this->offsets.template scan< Algorithms::ScanType::Exclusive >();
}

template< typename Device,
          typename Index >
__cuda_callable__
Index
CSR< Device, Index >::
getSegmentsCount() const
{
   return this->offsets.getSize() - 1;
}

template< typename Device,
          typename Index >
__cuda_callable__
Index
CSR< Device, Index >::
getSegmentSize( const IndexType segmentIdx ) const
{
   if( ! std::is_same< DeviceType, Devices::Host >::value )
   {
#ifdef __CUDA_ARCH__
      return offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ];
#else
      return offsets.getElement( segmentIdx + 1 ) - offsets.getElement( segmentIdx );
#endif
   }
   return offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ];
}

template< typename Device,
          typename Index >
__cuda_callable__
Index
CSR< Device, Index >::
getSize() const
{
   return this->getStorageSize();
}

template< typename Device,
          typename Index >
__cuda_callable__
Index
CSR< Device, Index >::
getStorageSize() const
{
   if( ! std::is_same< DeviceType, Devices::Host >::value )
   {
#ifdef __CUDA_ARCH__
      return offsets[ this->getSegmentsCount() ];
#else
      return offsets.getElement( this->getSegmentsCount() );
#endif
   }
   return offsets[ this->getSegmentsCount() ];
}

template< typename Device,
          typename Index >
__cuda_callable__
Index
CSR< Device, Index >::
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
          typename Index >
__cuda_callable__
void
CSR< Device, Index >::
getSegmentAndLocalIndex( const Index globalIdx, Index& segmentIdx, Index& localIdx ) const
{
}

template< typename Device,
          typename Index >
   template< typename Function, typename... Args >
void
CSR< Device, Index >::
forSegments( IndexType first, IndexType last, Function& f, Args... args ) const
{
   const auto offsetsView = this->offsets.getView();
   auto l = [=] __cuda_callable__ ( const IndexType i, Args... args ) mutable {
      const IndexType begin = offsetsView[ i ];
      const IndexType end = offsetsView[ i + 1 ];
      for( IndexType j = begin; j < end; j++  )
         if( ! f( i, j, args... ) )
            break;
   };
   Algorithms::ParallelFor< Device >::exec( first, last, l, args... );
}

template< typename Device,
          typename Index >
   template< typename Function, typename... Args >
void
CSR< Device, Index >::
forAll( Function& f, Args... args ) const
{
   this->forSegments( 0, this->getSize(), f, args... );
}

template< typename Device,
          typename Index >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
CSR< Device, Index >::
segmentsReduction( IndexType first, IndexType last, Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   using RealType = decltype( fetch( IndexType(), IndexType() ) );
   const auto offsetsView = this->offsets.getConstView();
   auto l = [=] __cuda_callable__ ( const IndexType i, Args... args ) mutable {
      const IndexType begin = offsetsView[ i ];
      const IndexType end = offsetsView[ i + 1 ];
      RealType aux( zero );
      for( IndexType j = begin; j < end; j++  )
         reduction( aux, fetch( i, j, args... ) );
      keeper( i, aux );
   };
   Algorithms::ParallelFor< Device >::exec( first, last, l, args... );
}

template< typename Device,
          typename Index >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
CSR< Device, Index >::
allReduction( Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->segmentsReduction( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index >
void
CSR< Device, Index >::
save( File& file ) const
{
   file << this->offsets;
}

template< typename Device,
          typename Index >
void
CSR< Device, Index >::
load( File& file )
{
   file >> this->offsets;
}

      } // namespace Segments
   }  // namespace Conatiners
} // namespace TNL
