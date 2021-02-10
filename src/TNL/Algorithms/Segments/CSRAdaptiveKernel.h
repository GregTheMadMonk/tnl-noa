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
#include <TNL/Algorithms/Segments/CSRAdaptiveKernelView.h>
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
struct CSRAdaptiveKernel
{
   using IndexType = Index;
   using DeviceType = Device;
   using ViewType = CSRAdaptiveKernelView< Index, Device >;
   using ConstViewType = CSRAdaptiveKernelView< Index, Device >;
   using BlocksType = typename ViewType::BlocksType;
   using BlocksView = typename BlocksType::ViewType;

   static constexpr int MaxValueSizeLog() { return ViewType::MaxValueSizeLog; };

   static TNL::String getKernelType();


   static constexpr Index THREADS_ADAPTIVE = details::CSRAdaptiveKernelParameters< sizeof( Index ) >::CudaBlockSize(); //sizeof(Index) == 8 ? 128 : 256;

   // How many shared memory use per block in CSR Adaptive kernel
   static constexpr Index SHARED_PER_BLOCK = details::CSRAdaptiveKernelParameters< sizeof( Index ) >::StreamedSharedMemory(); //20000; //24576; TODO:

   // Number of elements in shared memory 
   static constexpr Index SHARED = SHARED_PER_BLOCK/sizeof(double);

   // Number of warps in block for CSR Adaptive 
   static constexpr Index WARPS = THREADS_ADAPTIVE / 32;

   // Number of elements in shared memory per one warp 
   static constexpr Index SHARED_PER_WARP = SHARED / WARPS;

   // Max length of row to process one warp for CSR Light, MultiVector 
   static constexpr Index MAX_ELEMENTS_PER_WARP = 384;

   // Max length of row to process one warp for CSR Adaptive 
   static constexpr Index MAX_ELEMENTS_PER_WARP_ADAPT = details::CSRAdaptiveKernelParameters< sizeof( Index ) >::MaxAdaptiveElementsPerWarp();

   template< typename Offsets >
   void init( const Offsets& offsets );

   void reset();

   ViewType getView();

   ConstViewType getConstView() const;

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
                        Args... args ) const;

   protected:
      template< int SizeOfValue, typename Offsets >
      Index findLimit( const Index start,
                     const Offsets& offsets,
                     const Index size,
                     details::Type &type,
                     Index &sum );

      template< int SizeOfValue,
                typename Offsets >
      void initValueSize( const Offsets& offsets );

      /**
       * \brief  blocksArray[ i ] stores blocks for sizeof( Value ) == 2^i.
       */
      BlocksType blocksArray[ MaxValueSizeLog() ];

      ViewType view;
};

      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL

#include <TNL/Algorithms/Segments/CSRAdaptiveKernel.hpp>
