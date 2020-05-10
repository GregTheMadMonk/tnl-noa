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
#include <TNL/Containers/Segments/details/LambdaAdapter.h>

namespace TNL {
   namespace Containers {
      namespace Segments {


template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
__cuda_callable__
EllpackView< Device, Index, Organization, Alignment >::
EllpackView()
   : segmentSize( 0 ), size( 0 ), alignedSize( 0 )
{
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
__cuda_callable__
EllpackView< Device, Index, Organization, Alignment >::
EllpackView( IndexType segmentSize, IndexType size, IndexType alignedSize )
   : segmentSize( segmentSize ), size( size ), alignedSize( alignedSize )
{
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
__cuda_callable__
EllpackView< Device, Index, Organization, Alignment >::
EllpackView( const EllpackView& ellpack )
   : segmentSize( ellpack.segmentSize ), size( ellpack.size ), alignedSize( ellpack.alignedSize )
{
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
__cuda_callable__
EllpackView< Device, Index, Organization, Alignment >::
EllpackView( const EllpackView&& ellpack )
   : segmentSize( ellpack.segmentSize ), size( ellpack.size ), alignedSize( ellpack.alignedSize )
{
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
String
EllpackView< Device, Index, Organization, Alignment >::
getSerializationType()
{
   return "Ellpack< [any_device], " + TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
String
EllpackView< Device, Index, Organization, Alignment >::
getSegmentsType()
{
   return "Ellpack";
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
__cuda_callable__
typename EllpackView< Device, Index, Organization, Alignment >::ViewType
EllpackView< Device, Index, Organization, Alignment >::
getView()
{
   return ViewType( segmentSize, size, alignedSize );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
__cuda_callable__
auto
EllpackView< Device, Index, Organization, Alignment >::
getConstView() const -> const ConstViewType
{
   return ConstViewType( segmentSize, size, alignedSize );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
__cuda_callable__ auto EllpackView< Device, Index, Organization, Alignment >::
getSegmentsCount() const -> IndexType
{
   return this->size;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
__cuda_callable__ auto EllpackView< Device, Index, Organization, Alignment >::
getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   return this->segmentSize;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
__cuda_callable__ auto EllpackView< Device, Index, Organization, Alignment >::
getSize() const -> IndexType
{
   return this->size * this->segmentSize;
}


template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
__cuda_callable__ auto EllpackView< Device, Index, Organization, Alignment >::
getStorageSize() const -> IndexType
{
   return this->alignedSize * this->segmentSize;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
__cuda_callable__ auto EllpackView< Device, Index, Organization, Alignment >::
getGlobalIndex( const Index segmentIdx, const Index localIdx ) const -> IndexType
{
   if( Organization == RowMajorOrder )
      return segmentIdx * this->segmentSize + localIdx;
   else
      return segmentIdx + this->alignedSize * localIdx;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
__cuda_callable__ void EllpackView< Device, Index, Organization, Alignment >::
getSegmentAndLocalIndex( const Index globalIdx, Index& segmentIdx, Index& localIdx ) const
{
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
__cuda_callable__ auto EllpackView< Device, Index, Organization, Alignment >::
getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   if( Organization == RowMajorOrder )
      return SegmentViewType( segmentIdx * this->segmentSize, this->segmentSize, 1 );
   else
      return SegmentViewType( segmentIdx, this->segmentSize, this->alignedSize );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
   template< typename Function, typename... Args >
void EllpackView< Device, Index, Organization, Alignment >::
forSegments( IndexType first, IndexType last, Function& f, Args... args ) const
{
   if( Organization == RowMajorOrder )
   {
      const IndexType segmentSize = this->segmentSize;
      auto l = [=] __cuda_callable__ ( const IndexType segmentIdx, Args... args ) mutable {
         const IndexType begin = segmentIdx * segmentSize;
         const IndexType end = begin + segmentSize;
         IndexType localIdx( 0 );
         bool compute( true );
         for( IndexType globalIdx = begin; globalIdx < end && compute; globalIdx++  )
            f( segmentIdx, localIdx++, globalIdx, compute );
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
         bool compute( true );
         for( IndexType globalIdx = begin; globalIdx < end && compute; globalIdx += alignedSize )
            f( segmentIdx, localIdx++, globalIdx, compute, args... );
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l, args... );
   }
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
   template< typename Function, typename... Args >
void EllpackView< Device, Index, Organization, Alignment >::
forAll( Function& f, Args... args ) const
{
   this->forSegments( 0, this->getSegmentsCount(), f, args... );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void EllpackView< Device, Index, Organization, Alignment >::
segmentsReduction( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   //using RealType = decltype( fetch( IndexType(), IndexType(), IndexType(), std::declval< bool& >(), args... ) );
   using RealType = typename details::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   if( Organization == RowMajorOrder )
   {
      const IndexType segmentSize = this->segmentSize;
      auto l = [=] __cuda_callable__ ( const IndexType segmentIdx, Args... args ) mutable {
         const IndexType begin = segmentIdx * segmentSize;
         const IndexType end = begin + segmentSize;
         RealType aux( zero );
         IndexType localIdx( 0 );
         bool compute( true );
         for( IndexType j = begin; j < end && compute; j++  )
            aux = reduction( aux, details::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, j, compute ) );
         keeper( segmentIdx, aux );
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
         RealType aux( zero );
         IndexType localIdx( 0 );
         bool compute( true );
         for( IndexType j = begin; j < end && compute; j += alignedSize  )
            aux = reduction( aux, details::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, j, compute ) );
         keeper( segmentIdx, aux );
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l, args... );
   }
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void EllpackView< Device, Index, Organization, Alignment >::
allReduction( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   this->segmentsReduction( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero, args... );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
EllpackView< Device, Index, Organization, Alignment >&
EllpackView< Device, Index, Organization, Alignment >::
operator=( const EllpackView< Device, Index, Organization, Alignment >& view )
{
   this->segmentSize = view.segmentSize;
   this->size = view.size;
   this->alignedSize = view.alignedSize;
   return *this;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
void EllpackView< Device, Index, Organization, Alignment >::
save( File& file ) const
{
   file.save( &segmentSize );
   file.save( &size );
   file.save( &alignedSize );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
void EllpackView< Device, Index, Organization, Alignment >::
load( File& file )
{
   file.load( &segmentSize );
   file.load( &size );
   file.load( &alignedSize );
}

      } // namespace Segments
   }  // namespace Conatiners
} // namespace TNL
