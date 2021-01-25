/***************************************************************************
                          CSRKernels.h -  description
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
#include <TNL/Algorithms/Segments/CSRKernelScalar.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {

enum class Type {
   /* LONG = 0!!! Non zero value rewrites index[1] */
   LONG = 0,
   STREAM = 1,
   VECTOR = 2
};

/*template< typename Index >
struct LongBlockDescription
{
   uint8_t type;
}*/
template< typename Index >
union Block
{
   Block(Index row, Type type = Type::VECTOR, Index index = 0) noexcept
   {
      this->index[0] = row;
      this->index[1] = index;
      this->byte[sizeof(Index) == 4 ? 7 : 15] = (uint8_t)type;
   }

   Block(Index row, Type type, Index nextRow, Index maxID, Index minID) noexcept
   {
      this->index[0] = row;
      this->index[1] = 0;
      this->twobytes[sizeof(Index) == 4 ? 2 : 4] = maxID - minID;

      if (type == Type::STREAM)
         this->twobytes[sizeof(Index) == 4 ? 3 : 5] = nextRow - row;

      if (type == Type::STREAM)
         this->byte[sizeof(Index) == 4 ? 7 : 15] |= 0b1000000;
      else if (type == Type::VECTOR)
         this->byte[sizeof(Index) == 4 ? 7 : 15] |= 0b10000000;
   }

   Block() = default;

   __cuda_callable__ Type getType() const
   {
      if( byte[ sizeof( Index ) == 4 ? 7 : 15 ] & 0b1000000 )
         return Type::STREAM;
      if( byte[ sizeof( Index ) == 4 ? 7 : 15 ] & 0b10000000 )
         return Type::VECTOR;
      return Type::LONG;
   }

   __cuda_callable__ const Index& getFirstSegment() const
   {
      return index[ 0 ];
   }

   /***
    * \brief Returns number of elements covered by the block.
    */
   __cuda_callable__ const Index getSize() const
   {
      return twobytes[ sizeof(Index) == 4 ? 2 : 4 ];
   }

   /***
    * \brief Returns number of segments covered by the block.
    */
   __cuda_callable__ const Index getSegmentsInBlock() const
   {
      return ( twobytes[ sizeof( Index ) == 4 ? 3 : 5 ] & 0x3FFF );
   }

   void print( std::ostream& str ) const
   {
      Type type = this->getType();
      str << "Type: ";
      switch( type )
      {
         case Type::STREAM:
            str << " Stream ";
            break;
         case Type::VECTOR:
            str << " Vector ";
            break;
         case Type::LONG:
            str << " Long ";
            break;
      }
      str << " first segment: " << getFirstSegment();
      str << " block end: " << getSize();
      str << " index in warp: " << index[ 1 ];
   }
   Index index[2]; // index[0] is row pointer, index[1] is index in warp
   uint8_t byte[sizeof(Index) == 4 ? 8 : 16]; // byte[7/15] is type specificator
   uint16_t twobytes[sizeof(Index) == 4 ? 4 : 8]; //twobytes[2/4] is maxID - minID
                                                //twobytes[3/5] is nextRow - row
};

template< typename Index >
std::ostream& operator<< ( std::ostream& str, const Block< Index >& block )
{
   block.print( str );
   return str;
}

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
                                    Args... args )
{
   __shared__ Real streamShared[WARPS][SHARED_PER_WARP];
   __shared__ Real multivectorShared[ CudaBlockSize / warpSize ];
   constexpr size_t MAX_X_DIM = 2147483647;
   const Index index = (gridIdx * MAX_X_DIM) + (blockIdx.x * blockDim.x) + threadIdx.x;
   const Index blockIdx = index / warpSize;
   if( blockIdx >= blocks.getSize() - 1 )
      return;

   if( threadIdx.x < CudaBlockSize / warpSize )
      multivectorShared[ threadIdx.x ] = zero;
   Real result = zero;
   bool compute( true );
   const Index laneID = threadIdx.x & 31; // & is cheaper than %
   const Block< Index > block = blocks[ blockIdx ];
   const Index& firstSegmentIdx = block.getFirstSegment();
   const Index begin = offsets[ firstSegmentIdx ];
   //Index to, maxID;

   const auto blockType = block.getType();
   if( blockType == Type::STREAM )
   {
      const Index warpID = threadIdx.x / 32;
      const Index end = begin + block.getSize();

      // Stream data to shared memory
      for( Index globalIdx = laneID + begin; globalIdx < end; globalIdx += warpSize )
      {
         streamShared[warpID][globalIdx - begin ] = //fetch( globalIdx, compute );
            details::FetchLambdaAdapter< Index, Fetch >::call( fetch, -1, -1, globalIdx, compute );
         // TODO:: fix this by template specialization so that we can assume fetch lambda
         // with short parameters
      }

      const Index maxRow = firstSegmentIdx + block.getSegmentsInBlock();
      /* minRow */ //+
         /* maxRow - minRow *///(block.twobytes[sizeof(Index) == 4 ? 3 : 5] & 0x3FFF);
      /// Calculate result 
      for( Index i = block.index[0]/* minRow */ + laneID; i < maxRow; i += warpSize )
      {
         const Index to = offsets[i + 1] - begin; // end of preprocessed data
         result = zero;
         // Scalar reduction
         for( Index sharedID = offsets[ i ] - begin; sharedID < to; ++sharedID)
         {
            result = reduce( result, streamShared[warpID][sharedID] );
            //printf( " threadIdx %d is adding %d in segment %d -> %d\n", threadIdx.x, shared[warpID][sharedID], i, result );
         }

         //printf( "Stream: threadIdx = %d result for segment %d is %d \n", threadIdx.x, i, result );
         keep( i, result );
      }
   }
   else if( blockType == Type::VECTOR )
   {
      //printf( "Vector: threadIdx = %d \n", threadIdx );
      /////////////////////////////////////* CSR VECTOR *//////////////
      const Index end = begin + block.getSize(); //block.twobytes[sizeof(Index) == 4 ? 2 : 4];
      const Index segmentIdx = block.index[0];

      for( Index globalIdx = begin + laneID; globalIdx < end; globalIdx += warpSize )
         result = reduce( result, details::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, -1, globalIdx, compute ) ); // fix local idx

      /* Parallel reduction */
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 16 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result,  8 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result,  4 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result,  2 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result,  1 ) );
      if( laneID == 0 )
      {
         //printf( "Vector: threadIdx = %d result for segment %d is %f \n", threadIdx, segmentIdx, result );
         keep( segmentIdx, result );
          //outVector[block.index[0]/* minRow */] = result; // Write result
      }
   }
   else // blockType == Type::LONG
   {
      ///////////////////////////////////// CSR VECTOR L /////////////
      // Number of elements processed by previous warps
      const Index offset = block.index[1] * MAX_ELEM_PER_WARP;
      Index to = begin + (block.index[1]  + 1) * MAX_ELEM_PER_WARP;
      const Index segmentIdx = block.index[0];
      //minID = offsets[block.index[0] ];
      const Index end = offsets[block.index[0] + 1];
      const int tid = threadIdx.x;
      
      if( to > end )
         to = end;
      result = zero;
      //printf( "tid %d : start = %d \n", tid, minID + laneID );
      for( Index globalIdx = begin + laneID + offset; globalIdx < to; globalIdx += warpSize )
      {
         result = reduce( result, details::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, -1, globalIdx, compute ) );
         //printf( "tid %d -> %d \n", tid, details::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, -1, globalIdx, compute ) );
         //result += values[i] * inVector[columnIndexes[i]];
      }


      result += __shfl_down_sync(0xFFFFFFFF, result, 16);
      result += __shfl_down_sync(0xFFFFFFFF, result, 8);
      result += __shfl_down_sync(0xFFFFFFFF, result, 4);
      result += __shfl_down_sync(0xFFFFFFFF, result, 2);
      result += __shfl_down_sync(0xFFFFFFFF, result, 1);
      const Index warpID = threadIdx.x / 32;
      if( laneID == 0 )
         multivectorShared[ warpID ] = result;
      __syncthreads();
      // Reduction in multivectorShared
      if( tid < 16 )
      {
         multivectorShared[ tid ] =  reduce( multivectorShared[ tid ], multivectorShared[ tid + 16 ] );
         __syncwarp();
         multivectorShared[ tid ] =  reduce( multivectorShared[ tid ], multivectorShared[ tid +  8 ] );
         __syncwarp();
         multivectorShared[ tid ] =  reduce( multivectorShared[ tid ], multivectorShared[ tid +  4 ] );
         __syncwarp();
         multivectorShared[ tid ] =  reduce( multivectorShared[ tid ], multivectorShared[ tid +  2 ] );
         __syncwarp();
         multivectorShared[ tid ] =  reduce( multivectorShared[ tid ], multivectorShared[ tid +  1 ] );
         __syncwarp();
         if( tid == 0 )
         {
            printf( "Long: segmentIdx %d -> %d \n", segmentIdx, multivectorShared[ 0 ] );
            keep( segmentIdx, multivectorShared[ 0 ] );
         }
      }

      //if (laneID == 0) atomicAdd(&outVector[block.index[0] ], result);
   }
}
#endif


template< typename Index,
          typename Device >
struct CSRKernelAdaptiveView
{
   using IndexType = Index;
   using DeviceType = Device;
   using ViewType = CSRKernelAdaptiveView< Index, Device >;
   using ConstViewType = CSRKernelAdaptiveView< Index, Device >;
   using BlocksType = TNL::Containers::Vector< Block< Index >, Device, Index >;
   using BlocksView = typename BlocksType::ViewType;

   CSRKernelAdaptiveView() = default;

   CSRKernelAdaptiveView( BlocksType& blocks )
   {
      this->blocks.bind( blocks );
   };

   void setBlocks( BlocksType& blocks )
   {
      this->blocks.bind( blocks );
   }

   ViewType getView() { return *this; };

   ConstViewType getConstView() const { return *this; };

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
         TNL::Algorithms::Segments::CSRKernelScalar< Index, Device >::
            segmentsReduction( offsets, first, last, fetch, reduction, keeper, zero, args... );
         return;
      }

      //this->printBlocks();
      static constexpr Index THREADS_ADAPTIVE = sizeof(Index) == 8 ? 128 : 256;
      //static constexpr Index THREADS_SCALAR = 128;
      //static constexpr Index THREADS_VECTOR = 128;
      static constexpr Index THREADS_LIGHT = 128;

      /* Max length of row to process one warp for CSR Light, MultiVector */
      static constexpr Index MAX_ELEMENTS_PER_WARP = 384;

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

   CSRKernelAdaptiveView& operator=( const CSRKernelAdaptiveView< Index, Device >& kernelView )
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
struct CSRKernelAdaptive
{
    using IndexType = Index;
    using DeviceType = Device;
    using ViewType = CSRKernelAdaptiveView< Index, Device >;
    using ConstViewType = CSRKernelAdaptiveView< Index, Device >;
    using BlocksType = typename ViewType::BlocksType;
    using BlocksView = typename BlocksType::ViewType;


    static constexpr Index THREADS_ADAPTIVE = sizeof(Index) == 8 ? 128 : 256;

   /* How many shared memory use per block in CSR Adaptive kernel */
   static constexpr Index SHARED_PER_BLOCK = 24576;

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
                    Type &type,
                    Index &sum )
   {
      sum = 0;
      for (Index current = start; current < size - 1; current++ )
      {
         Index elements = offsets.getElement(current + 1) -
                           offsets.getElement(current);
         sum += elements;
         if( sum > SHARED_PER_WARP )
         {
            if( current - start > 0 ) // extra row
            {
               type = Type::STREAM;
               return current;
            }
            else
            {                  // one long row
               if( sum <= 2 * MAX_ELEMENTS_PER_WARP_ADAPT )
                  type = Type::VECTOR;
               else
                  type = Type::VECTOR; // TODO: Put LONG back
                  //type = Type::LONG; //
               return current + 1;
            }
         }
      }
      type = Type::STREAM;
      return size - 1; // return last row pointer
    }

    template< typename Offsets >
    void init( const Offsets& offsets )
    {
        const Index rows = offsets.getSize();
        Index sum, start( 0 ), nextStart( 0 );

        // Fill blocks
        std::vector< Block< Index > > inBlock;
        inBlock.reserve( rows );

        while( nextStart != rows - 1 )
        {
            Type type;
            nextStart = findLimit( start, offsets, rows, type, sum );

            if( type == Type::LONG )
            {
               inBlock.emplace_back( start, Type::LONG, 0 );
               const Index blocksCount = inBlock.size();
               const Index warpsPerCudaBlock = THREADS_ADAPTIVE / TNL::Cuda::getWarpSize();
               const Index warpsLeft = roundUpDivision( blocksCount, warpsPerCudaBlock ) * warpsPerCudaBlock - blocksCount;
               //Index parts = roundUpDivision(sum, this->SHARED_PER_WARP);
               /*for( Index index = 1; index < warpsLeft; index++ )
               {
                  inBlock.emplace_back(start, Type::LONG, index);
               }*/
            }
            else
            {
               inBlock.emplace_back(start, type,
                    nextStart,
                    offsets.getElement(nextStart),
                    offsets.getElement(start) );
            }
            start = nextStart;
        }
        inBlock.emplace_back(nextStart);

        // Copy values
        this->blocks.setSize(inBlock.size());
        for (size_t i = 0; i < inBlock.size(); ++i)
            this->blocks.setElement(i, inBlock[i]);

         this->view.setBlocks( blocks );
    };

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
