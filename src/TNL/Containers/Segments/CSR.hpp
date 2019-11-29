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
#include <TNL/Algorithms/ParalleFor.h>
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
CSR< Device, Index >::
void setSegmentsCount( const IndexType& size )
{
   this->offsets.setSize( size + 1 );
}

template< typename Device,
          typename Index >
   template< typename SizesHolder = OffsetsHolder >
CSR< Device, Index >::
void setSizes( const SizesHolder& sizes )
{
   this->offsets.setSize( sizes.getSize() + 1 );
   auto view = this->offsets.getView( 0, sizes.getSize() );
   view = sizes;
   this->offsets.setElement( sizes.getSize>(), 0 );
   this->offsets.scan< Algorithms::ScanType::Exclusive >();
}

template< typename Device,
          typename Index >
CSR< Device, Index >::
Index getSize() const
{
   return this->offsets.getSize() - 1;
}

template< typename Device,
          typename Index >
   template< typename Function, typename... Args >
CSR< Device, Index >::
void forAll( Function& f, Args args ) const
{
   const auto offsetsView = this->offsets.getView();
   auto f = [=] __cuda_callable__ ( const IndexType i, f, args ) {
      const IndexType begin = offsetsView[ i ];
      const IndexType end = offsetsView[ i + 1 ];
      for( IndexType j = begin; j < end; j++  )
         f( i, j, args );
   };
   Algorithms::ParallelFor< Device >::exec( 0, this->getSize(), f );
}

template< typename Device,
          typename Index >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
CSR< Device, Index >::
void segmentsReduction( Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, Real zero, Args args )
{
   const auto offsetsView = this->offsets.getView();
   auto f = [=] __cuda_callable__ ( const IndexType i, f, args ) {
      const IndexType begin = offsetsView[ i ];
      const IndexType end = offsetsView[ i + 1 ];
      Real aux( zero );
      for( IndexType j = begin; j < end; j++  )
         reduction( aux, fetch( i, j, args ) );
      keeper( i, aux );
   };
   Algorithms::ParallelFor< Device >::exec( 0, this->getSize(), f );
}


      } // namespace Segements
   }  // namespace Conatiners
} // namespace TNL