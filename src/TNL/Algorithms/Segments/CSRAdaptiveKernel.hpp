/***************************************************************************
                          CSRAdaptiveKernel.hpp -  description
                             -------------------
    begin                : Feb 7, 2021 -> Joe Biden inauguration
    copyright            : (C) 2021 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>
#include <TNL/Cuda/LaunchHelpers.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Algorithms/Segments/details/LambdaAdapter.h>
#include <TNL/Algorithms/Segments/CSRScalarKernel.h>
#include <TNL/Algorithms/Segments/details/CSRAdaptiveKernelBlockDescriptor.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {

template< typename Index,
          typename Device >
TNL::String
CSRAdaptiveKernel< Index, Device >::
getKernelType()
{
   return ViewType::getKernelType();
};

template< typename Index,
          typename Device >
   template< typename Offsets >
void
CSRAdaptiveKernel< Index, Device >::
init( const Offsets& offsets )
{
   this->template initValueSize<  1 >( offsets );
   this->template initValueSize<  2 >( offsets );
   this->template initValueSize<  4 >( offsets );
   this->template initValueSize<  8 >( offsets );
   this->template initValueSize< 16 >( offsets );
   this->template initValueSize< 32 >( offsets );
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      this->view.setBlocks( blocksArray[ i ], i );
}


template< typename Index,
          typename Device >
void
CSRAdaptiveKernel< Index, Device >::
reset()
{
   for( int i = 0; i < MaxValueSizeLog(); i++ )
   {
      this->blocksArray[ i ].reset();
      this->view.setBlocks( this->blocksArray[ i ], i );
   }
}

template< typename Index,
          typename Device >
auto
CSRAdaptiveKernel< Index, Device >::
getView() -> ViewType
{
   return this->view;
}

template< typename Index,
          typename Device >
auto
CSRAdaptiveKernel< Index, Device >::
getConstView() const -> ConstViewType
{
   return this->view;
};

template< typename Index,
          typename Device >
   template< typename OffsetsView,
               typename Fetch,
               typename Reduction,
               typename ResultKeeper,
               typename Real,
               typename... Args >
void
CSRAdaptiveKernel< Index, Device >::
segmentsReduction( const OffsetsView& offsets,
                   Index first,
                   Index last,
                   Fetch& fetch,
                   const Reduction& reduction,
                   ResultKeeper& keeper,
                   const Real& zero,
                   Args... args ) const
{
   view.segmentsReduction( offsets, first, last, fetch, reduction, keeper, zero, args... );
}

template< typename Index,
          typename Device >
   template< int SizeOfValue,
             typename Offsets >
Index
CSRAdaptiveKernel< Index, Device >::
findLimit( const Index start,
           const Offsets& offsets,
           const Index size,
           details::Type &type,
           Index &sum )
{
   sum = 0;
   for (Index current = start; current < size - 1; current++ )
   {
      Index elements = offsets[ current + 1 ] - offsets[ current ];
      sum += elements;
      if( sum > details::CSRAdaptiveKernelParameters< SizeOfValue >::StreamedSharedElementsPerWarp() )
      {
         if( current - start > 0 ) // extra row
         {
            type = details::Type::STREAM;
            return current;
         }
         else
         {                  // one long row
            if( sum <= 2 * details::CSRAdaptiveKernelParameters< SizeOfValue >::MaxAdaptiveElementsPerWarp() ) //MAX_ELEMENTS_PER_WARP_ADAPT )
               type = details::Type::VECTOR;
            else
               type = details::Type::LONG;
            return current + 1;
         }
      }
   }
   type = details::Type::STREAM;
   return size - 1; // return last row pointer
}

template< typename Index,
          typename Device >
   template< int SizeOfValue,
             typename Offsets >
void
CSRAdaptiveKernel< Index, Device >::
initValueSize( const Offsets& offsets )
{
   using HostOffsetsType = TNL::Containers::Vector< typename Offsets::IndexType, TNL::Devices::Host, typename Offsets::IndexType >;
   HostOffsetsType hostOffsets( offsets );
   const Index rows = offsets.getSize();
   Index sum, start( 0 ), nextStart( 0 );

   // Fill blocks
   std::vector< details::CSRAdaptiveKernelBlockDescriptor< Index > > inBlocks;
   inBlocks.reserve( rows );

   while( nextStart != rows - 1 )
   {
      details::Type type;
      nextStart = findLimit< SizeOfValue >( start, hostOffsets, rows, type, sum );

      if( type == details::Type::LONG )
      {
         const Index blocksCount = inBlocks.size();
         const Index warpsPerCudaBlock = details::CSRAdaptiveKernelParameters< sizeof( Index ) >::CudaBlockSize() / TNL::Cuda::getWarpSize();
         Index warpsLeft = roundUpDivision( blocksCount, warpsPerCudaBlock ) * warpsPerCudaBlock - blocksCount;
         if( warpsLeft == 0 )
            warpsLeft = warpsPerCudaBlock;
         for( Index index = 0; index < warpsLeft; index++ )
            inBlocks.emplace_back( start, details::Type::LONG, index, warpsLeft );
      }
      else
      {
         inBlocks.emplace_back(start, type,
               nextStart,
               offsets.getElement(nextStart),
               offsets.getElement(start) );
      }
      start = nextStart;
   }
   inBlocks.emplace_back(nextStart);
   //std::cerr << "Setting blocks to " << std::log2( SizeOfValue ) << std::endl;
   TNL_ASSERT_LT( std::log2( SizeOfValue ), MaxValueSizeLog(), "" );
   TNL_ASSERT_GE( std::log2( SizeOfValue ), 0, "" );
   this->blocksArray[ (int ) std::log2( SizeOfValue ) ] = inBlocks;
}

      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL
