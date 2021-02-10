/***************************************************************************
                          CSRAdaptiveKernelBlockDescriptor.h -  description
                             -------------------
    begin                : Jan 25, 2021 -> Joe Biden inauguration
    copyright            : (C) 2021 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
   namespace Algorithms {
      namespace Segments {
         namespace details {

template< int SizeOfValue,
          int StreamedSharedMemory_ = 24576 >
struct CSRAdaptiveKernelParameters
{
   /**
    * \brief Computes number of CUDA threads per block depending on Value type.
    *
    * \return CUDA block size.
    */
   static constexpr int CudaBlockSize() { return 256; }; //sizeof( Value ) == 8 ? 128 : 256; };
    //std::max( ( int ) ( 1024 / sizeof( Value ) ), ( int ) Cuda::getWarpSize() ); };

   /**
    * \brief Returns amount of shared memory dedicated for stream CSR kernel.
    *
    * \return Stream shared memory.
    */
   static constexpr size_t StreamedSharedMemory() { return StreamedSharedMemory_; };

   /**
    * \brief Number of elements fitting into streamed shared memory.
    */
   static constexpr size_t StreamedSharedElementsCount() { return StreamedSharedMemory() / SizeOfValue; };

   /**
    * \brief Computes number of warps in one CUDA block.
    */
   static constexpr size_t WarpsCount() { return CudaBlockSize() / Cuda::getWarpSize(); };

   /**
    * \brief Computes number of elements to be streamed into the shared memory.
    *
    * \return Number of elements to be streamed into the shared memory.
    */
   static constexpr size_t StreamedSharedElementsPerWarp() { return StreamedSharedElementsCount() / WarpsCount(); };

   /**
    * \brief Returns maximum number of elements per warp for vector and hybrid kernel.
    *
    * \return Maximum number of elements per warp for vector and hybrid kernel.
    */
   static constexpr int MaxVectorElementsPerWarp() { return 384; };

   /**
    * \brief Returns maximum number of elements per warp for adaptive kernel.
    *
    * \return Maximum number of elements per warp for adaptive kernel.
    */
   static constexpr int MaxAdaptiveElementsPerWarp() { return 512; };
};

         } // namespace details
      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL
