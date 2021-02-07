/***************************************************************************
                          CSRAdaptiveKernel.h -  description
                             -------------------
    begin                : Jan 20, 2021 -> Joe Biden inauguration
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

#ifdef HAVE_CUDA

template< int CudaBlockSize,
          int warpSize,
          int WARPS,
          int SHARED_PER_WARP,
          int MAX_ELEM_PER_WARP,
          typename BlocksView,
          typename Offsets,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Real,
          typename... Args >
__global__ void
segmentsReductionCSRAdaptiveKernel( BlocksView blocks,
                                    int gridIdx,
                                    Offsets offsets,
                                    Index first,
                                    Index last,
                                    Fetch fetch,
                                    Reduction reduce,
                                    ResultKeeper keep,
                                    Real zero,
                                    Args... args );
#endif

template< typename Index,
          typename Device >
struct CSRAdaptiveKernelView
{
   using IndexType = Index;
   using DeviceType = Device;
   using ViewType = CSRAdaptiveKernelView< Index, Device >;
   using ConstViewType = CSRAdaptiveKernelView< Index, Device >;
   using BlocksType = TNL::Containers::Vector< details::CSRAdaptiveKernelBlockDescriptor< Index >, Device, Index >;
   using BlocksView = typename BlocksType::ViewType;

   CSRAdaptiveKernelView() = default;

   CSRAdaptiveKernelView( BlocksType& blocks )
   {
      this->blocks.bind( blocks );
   };

   void setBlocks( BlocksType& blocks )
   {
      this->blocks.bind( blocks );
   }

   ViewType getView() { return *this; };

   ConstViewType getConstView() const { return *this; };

   static TNL::String getKernelType()
   {
      return "Adaptive";
   };

   template< typename OffsetsView,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Real,
             typename... Args >
   void segmentsReduction( const OffsetsView& offsets,
                        Index first,
                        Index last,
                        Fetch& fetch,
                        const Reduction& reduction,
                        ResultKeeper& keeper,
                        const Real& zero,
                        Args... args ) const
   {
#ifdef HAVE_CUDA
      if( details::CheckFetchLambda< Index, Fetch >::hasAllParameters() )
      {
         TNL::Algorithms::Segments::CSRScalarKernel< Index, Device >::
            segmentsReduction( offsets, first, last, fetch, reduction, keeper, zero, args... );
         return;
      }

      static constexpr Index THREADS_ADAPTIVE = sizeof(Index) == 8 ? 128 : 256;
      //static constexpr Index THREADS_SCALAR = 128;
      //static constexpr Index THREADS_VECTOR = 128;
      //static constexpr Index THREADS_LIGHT = 128;

      /* Max length of row to process one warp for CSR Light, MultiVector */
      //static constexpr Index MAX_ELEMENTS_PER_WARP = 384;

      /* Max length of row to process one warp for CSR Adaptive */
      static constexpr Index MAX_ELEMENTS_PER_WARP_ADAPT = 512;

      /* How many shared memory use per block in CSR Adaptive kernel */
      static constexpr Index SHARED_PER_BLOCK = 24576;

      /* Number of elements in shared memory */
      static constexpr Index SHARED = SHARED_PER_BLOCK/sizeof(Real);

      /* Number of warps in block for CSR Adaptive */
      static constexpr Index WARPS = THREADS_ADAPTIVE / 32;

      /* Number of elements in shared memory per one warp */
      static constexpr Index SHARED_PER_WARP = SHARED / WARPS;

      constexpr int warpSize = 32;

      Index blocksCount;

      const Index threads = THREADS_ADAPTIVE;
      constexpr size_t MAX_X_DIM = 2147483647;

      /* Fill blocks */
      size_t neededThreads = this->blocks.getSize() * warpSize; // one warp per block
      /* Execute kernels on device */
      for (Index gridIdx = 0; neededThreads != 0; gridIdx++ )
      {
         if (MAX_X_DIM * threads >= neededThreads)
         {
            blocksCount = roundUpDivision(neededThreads, threads);
            neededThreads = 0;
         }
         else
         {
            blocksCount = MAX_X_DIM;
            neededThreads -= MAX_X_DIM * threads;
         }

         segmentsReductionCSRAdaptiveKernel<
               THREADS_ADAPTIVE,
               warpSize,
               WARPS,
               SHARED_PER_WARP,
               MAX_ELEMENTS_PER_WARP_ADAPT,
               BlocksView,
               OffsetsView,
               Index, Fetch, Reduction, ResultKeeper, Real, Args... >
            <<<blocksCount, threads>>>(
               this->blocks,
               gridIdx,
               offsets,
               first,
               last,
               fetch,
               reduction,
               keeper,
               zero,
               args... );
      }
#endif
   }

   CSRAdaptiveKernelView& operator=( const CSRAdaptiveKernelView< Index, Device >& kernelView )
   {
      this->blocks.bind( kernelView.blocks );
      return *this;
   }

   void printBlocks() const
   {
      for( Index i = 0; i < this->blocks.getSize(); i++ )
      {
         auto block = blocks.getElement( i );
         std::cout << "Block " << i << " : " << block << std::endl;
      }

   }

   protected:
      BlocksView blocks;
};

template< typename Index,
          typename Device >
struct CSRAdaptiveKernel
{
   using IndexType = Index;
   using DeviceType = Device;
   using ViewType = CSRAdaptiveKernelView< Index, Device >;
   using ConstViewType = CSRAdaptiveKernelView< Index, Device >;
   using BlocksType = typename ViewType::BlocksType;
   using BlocksView = typename BlocksType::ViewType;

   static TNL::String getKernelType()
   {
      return ViewType::getKernelType();
   };

    static constexpr Index THREADS_ADAPTIVE = sizeof(Index) == 8 ? 128 : 256;

   /* How many shared memory use per block in CSR Adaptive kernel */
   static constexpr Index SHARED_PER_BLOCK = 20000; //24576; TODO:

   /* Number of elements in shared memory */
   static constexpr Index SHARED = SHARED_PER_BLOCK/sizeof(double);

   /* Number of warps in block for CSR Adaptive */
   static constexpr Index WARPS = THREADS_ADAPTIVE / 32;

   /* Number of elements in shared memory per one warp */
   static constexpr Index SHARED_PER_WARP = SHARED / WARPS;

   /* Max length of row to process one warp for CSR Light, MultiVector */
   static constexpr Index MAX_ELEMENTS_PER_WARP = 384;

   /* Max length of row to process one warp for CSR Adaptive */
   static constexpr Index MAX_ELEMENTS_PER_WARP_ADAPT = 512;

   template< typename Offsets >
   Index findLimit( const Index start,
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
         if( sum > SHARED_PER_WARP )
         {
            if( current - start > 0 ) // extra row
            {
               type = details::Type::STREAM;
               return current;
            }
            else
            {                  // one long row
               if( sum <= 2 * MAX_ELEMENTS_PER_WARP_ADAPT )
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

   template< typename Offsets >
   void init( const Offsets& offsets )
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
         nextStart = findLimit( start, hostOffsets, rows, type, sum );

         if( type == details::Type::LONG )
         {
            const Index blocksCount = inBlocks.size();
            const Index warpsPerCudaBlock = THREADS_ADAPTIVE / TNL::Cuda::getWarpSize();
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
      this->blocks = inBlocks;
      this->view.setBlocks( blocks );
   }

   void reset()
   {
      this->blocks.reset();
      this->view.setBlocks( blocks );
   }

   ViewType getView() { return this->view; };

   ConstViewType getConstView() const { return this->view; };

   template< typename OffsetsView,
              typename Fetch,
              typename Reduction,
              typename ResultKeeper,
              typename Real,
              typename... Args >
   void segmentsReduction( const OffsetsView& offsets,
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

   protected:
      BlocksType blocks;

      ViewType view;
};

      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL

#include <TNL/Algorithms/Segments/CSRAdaptiveKernel.hpp>
