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
          typename Index >
Ellpack< Device, Index >::
Ellpack() : size( 0 ), rowLength( 0 )
{
}

template< typename Device,
          typename Index >
Ellpack< Device, Index >::
Ellpack( const Ellpack& ellpack ) : offsets( ellpack.offsets )
{
}

template< typename Device,
          typename Index >
Ellpack< Device, Index >::
Ellpack( const Ellpack&& ellpack ) : offsets( std::move( ellpack.offsets ) )
{

}

template< typename Device,
          typename Index >
   template< typename SizesHolder >
void
Ellpack< Device, Index >::
setSizes( const SizesHolder& sizes )
{
   this->segmentSize = max( sizes );
   this->size = sizes.getSize();
}

template< typename Device,
          typename Index >
__cuda_callable__
Index
Ellpack< Device, Index >::
getSize() const
{
   return this->offsets.getSize() - 1;
}

template< typename Device,
          typename Index >
__cuda_callable__
Index
Ellpack< Device, Index >::
getSegmentSize( const IndexType segmentIdx ) const
{
   return this->segmentSize;
}

template< typename Device,
          typename Index >
__cuda_callable__
Index
Ellpack< Device, Index >::
getStorageSize() const
{
   return this->size * this->segmentSize;
}

template< typename Device,
          typename Index >
__cuda_callable__
Index
Ellpack< Device, Index >::
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
Ellpack< Device, Index >::
getSegmentAndLocalIndex( const Index globalIdx, Index& segmentIdx, Index& localIdx ) const
{
}

template< typename Device,
          typename Index >
   template< typename Function, typename... Args >
void
Ellpack< Device, Index >::
forSegments( IndexType first, IndexType last, Function& f, Args... args ) const
{
   const auto offsetsView = this->offsets.getView();
   auto l = [=] __cuda_callable__ ( const IndexType i, Args... args ) {
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
Ellpack< Device, Index >::
forAll( Function& f, Args... args ) const
{
   this->forSegments( 0, this->getSize(), f, args... );
}

template< typename Device,
          typename Index >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void
Ellpack< Device, Index >::
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
Ellpack< Device, Index >::
allReduction( Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->segmentsReduction( 0, this->getSize(), fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index >
void
Ellpack< Device, Index >::
save( File& file ) const
{
   file << this->offsets;
}

template< typename Device,
          typename Index >
void
Ellpack< Device, Index >::
load( File& file )
{
   file >> this->offsets;
}

      } // namespace Segements
   }  // namespace Conatiners
} // namespace TNL
