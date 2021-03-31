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
#include <TNL/Algorithms/Segments/EllpackView.h>
#include <TNL/Algorithms/Segments/detail/LambdaAdapter.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {


template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
__cuda_callable__
EllpackView< Device, Index, Organization, Alignment >::
EllpackView()
   : segmentSize( 0 ), segmentsCount( 0 ), alignedSize( 0 )
{
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
__cuda_callable__
EllpackView< Device, Index, Organization, Alignment >::
EllpackView( IndexType segmentsCount, IndexType segmentSize, IndexType alignedSize )
   : segmentSize( segmentSize ), segmentsCount( segmentsCount ), alignedSize( alignedSize )
{
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
__cuda_callable__
EllpackView< Device, Index, Organization, Alignment >::
EllpackView( IndexType segmentsCount, IndexType segmentSize )
   : segmentSize( segmentSize ), segmentsCount( segmentsCount )
{
   if( Organization == RowMajorOrder )
      this->alignedSize = this->segmentsCount;
   else
      this->alignedSize = roundUpDivision( segmentsCount, this->getAlignment() ) * this->getAlignment();
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
__cuda_callable__
EllpackView< Device, Index, Organization, Alignment >::
EllpackView( const EllpackView& ellpack )
   : segmentSize( ellpack.segmentSize ), segmentsCount( ellpack.segmentsCount ), alignedSize( ellpack.alignedSize )
{
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
__cuda_callable__
EllpackView< Device, Index, Organization, Alignment >::
EllpackView( const EllpackView&& ellpack )
   : segmentSize( ellpack.segmentSize ), segmentsCount( ellpack.segmentsCount ), alignedSize( ellpack.alignedSize )
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
   return ViewType( segmentSize, segmentsCount, alignedSize );
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
   return ConstViewType( segmentsCount, segmentSize, alignedSize );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
__cuda_callable__ auto EllpackView< Device, Index, Organization, Alignment >::
getSegmentsCount() const -> IndexType
{
   return this->segmentsCount;
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
   return this->segmentsCount * this->segmentSize;
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
__cuda_callable__ auto EllpackView< Device, Index, Organization, Alignment >::
getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   if( Organization == RowMajorOrder )
      return SegmentViewType( segmentIdx, segmentIdx * this->segmentSize, this->segmentSize, 1 );
   else
      return SegmentViewType( segmentIdx, segmentIdx, this->segmentSize, this->alignedSize );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
   template< typename Function >
void EllpackView< Device, Index, Organization, Alignment >::
forElements( IndexType first, IndexType last, Function&& f ) const
{
   if( Organization == RowMajorOrder )
   {
      const IndexType segmentSize = this->segmentSize;
      auto l = [=] __cuda_callable__ ( const IndexType segmentIdx ) mutable {
         const IndexType begin = segmentIdx * segmentSize;
         const IndexType end = begin + segmentSize;
         IndexType localIdx( 0 );
         bool compute( true );
         for( IndexType globalIdx = begin; globalIdx < end && compute; globalIdx++  )
            f( segmentIdx, localIdx++, globalIdx, compute );
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l );
   }
   else
   {
      const IndexType storageSize = this->getStorageSize();
      const IndexType alignedSize = this->alignedSize;
      auto l = [=] __cuda_callable__ ( const IndexType segmentIdx ) mutable {
         const IndexType begin = segmentIdx;
         const IndexType end = storageSize;
         IndexType localIdx( 0 );
         bool compute( true );
         for( IndexType globalIdx = begin; globalIdx < end && compute; globalIdx += alignedSize )
            f( segmentIdx, localIdx++, globalIdx, compute );
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l );
   }
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
   template< typename Function >
void EllpackView< Device, Index, Organization, Alignment >::
forAllElements( Function&& f ) const
{
   this->forElements( 0, this->getSegmentsCount(), f );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
   template< typename Function >
void EllpackView< Device, Index, Organization, Alignment >::
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
          ElementsOrganization Organization,
          int Alignment >
   template< typename Function >
void EllpackView< Device, Index, Organization, Alignment >::
forAllSegments( Function&& f ) const
{
   this->forSegments( 0, this->getSegmentsCount(), f );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
void EllpackView< Device, Index, Organization, Alignment >::
reduceSegments( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const
{
   //using RealType = decltype( fetch( IndexType(), IndexType(), IndexType(), std::declval< bool& >(), args... ) );
   using RealType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
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
            aux = reduction( aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, j, compute ) );
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
            aux = reduction( aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, j, compute ) );
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
   this->reduceSegments( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero, args... );
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
   this->segmentsCount = view.segmentsCount;
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
   file.save( &segmentsCount );
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
   file.load( &segmentsCount );
   file.load( &alignedSize );
}

      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL
