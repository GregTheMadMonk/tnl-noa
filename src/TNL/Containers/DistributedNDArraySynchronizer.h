// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <future>
// 3rd-party async library providing a thread-pool
#include <async/threadpool.h>

#include <TNL/Containers/ndarray/SynchronizerBuffers.h>
#include <TNL/MPI/Comm.h>
#include <TNL/MPI/Wrappers.h>
#include <TNL/Timer.h>

namespace TNL {
namespace Containers {

enum class SyncDirection : std::uint8_t
{
   // special - sync in all directions
   All = 0xff,
   // sync directions like in LBM
   None = 0,
   Right = 1 << 0,
   Left = 1 << 1,

   // TODO: for 2D distribution:
   // Right = 1 << 0,
   // Left = 1 << 1,
   // Top = 1 << 2,
   // Bottom = 1 << 3,
   // TopRight = Top | Right,
   // TopLeft = Top | Left,
   // BottomRight = Bottom | Right,
   // BottomLeft = Bottom | Left

   // TODO: for 3D distribution:
   // Right = 1 << 0,
   // Left = 1 << 1,
   // Top = 1 << 2,
   // Bottom = 1 << 3,
   // Back = 1 << 4,
   // Front = 1 << 5,
   // TopRight = Top | Right,
   // TopLeft = Top | Left,
   // BottomRight = Bottom | Right,
   // BottomLeft = Bottom | Left
   // BackRight = Back | Right,
   // BackLeft = Back | Left,
   // FrontRight = Front | Right,
   // FrontLeft = Front | Left,
   // BackTop = Back | Top,
   // BackBottom = Back | Bottom,
   // FrontTop = Front | Top,
   // FrontBottom = Front | Bottom,
   // BackTopRight = Back | Top | Right,
   // BackTopLeft = Back | Top | Left,
   // BackBottomRight = Back | Bottom | Right,
   // BackBottomLeft = Back | Bottom | Left,
   // FrontTopRight = Front | Top | Right,
   // FrontTopLeft = Front | Top | Left,
   // FrontBottomRight = Front | Bottom | Right,
   // FrontBottomLeft = Front | Bottom | Left,
};

inline SyncDirection
operator&( SyncDirection a, SyncDirection b )
{
   return static_cast< SyncDirection >( static_cast< std::uint8_t >( a ) & static_cast< std::uint8_t >( b ) );
}

inline SyncDirection
operator|( SyncDirection a, SyncDirection b )
{
   return static_cast< SyncDirection >( static_cast< std::uint8_t >( a ) | static_cast< std::uint8_t >( b ) );
}

// this operator makes `a -= b` equivalent to `a &= ~b`, i.e. it clears all bits from b in a
inline SyncDirection&
operator-=( SyncDirection& a, SyncDirection b )
{
   a = static_cast< SyncDirection >( static_cast< std::uint8_t >( a ) & ~static_cast< std::uint8_t >( b ) );
   return a;
}

template< typename DistributedNDArray,
          // This can be set to false to optimize out buffering when it is not needed
          // (e.g. for LBM with 1D distribution and specific orientation of the ndarray)
          bool buffered = true >
class DistributedNDArraySynchronizer
{
private:
   // NOTE: async::threadpool has alignment requirements, which causes problems:
   //  - it may become misaligned in derived classes, see e.g.
   //    https://stackoverflow.com/a/46475498
   //    solution: specify it as the first member of the base class
   //  - operator new before C++17 may not support over-aligned types, see
   //    https://stackoverflow.com/a/53485295
   //    solution: relaxed alignment requirements to not exceed the value of
   //    alignof(std::max_align_t), which is the strongest alignment supported
   //    by plain new. See https://github.com/d36u9/async/pull/2
   async::threadpool tp;

   int gpu_id = 0;

   int tag_offset = 0;

   int tag_from_left = -1;
   int tag_from_right = -1;
   int tag_to_left = -1;
   int tag_to_right = -1;

   static int
   reserve_tags( int count )
   {
      static int offset = 0;
      // we could use a post-increment, but we don't have to start from 0 either...
      return offset += count;
   }

public:
   using RequestsVector = std::vector< MPI_Request >;

   enum class AsyncPolicy
   {
      synchronous,
      deferred,
      threadpool,
      async,
   };

   //   DistributedNDArraySynchronizer(int max_threads = std::thread::hardware_concurrency())
   DistributedNDArraySynchronizer( int max_threads = 1 )
   : tp( max_threads ), tag_offset( reserve_tags( 2 ) )  // reserve 2 communication tags (for left and right)
   {}

   // async::threadpool is not move-constructible (due to std::atomic), so we need
   // custom move-constructor that skips moving tp
   DistributedNDArraySynchronizer( DistributedNDArraySynchronizer&& other )
   : tp( other.tp.size() ), gpu_id( std::move( other.gpu_id ) ), tag_offset( std::move( other.tag_offset ) )
   {}

   void
   setTagOffset( int offset )
   {
      tag_offset = offset;
   }

   void
   setTags( int from_left_id, int to_left_id, int from_right_id, int to_right_id )
   {
      tag_from_left = from_left_id;
      tag_to_left = to_left_id;
      tag_from_right = from_right_id;
      tag_to_right = to_right_id;
   }

   // special thing for the A-A pattern in LBM
   template< std::size_t dim >
   void
   setBuffersShift( int shift )
   {
      auto& dim_buffers = buffers.template getDimBuffers< dim >();

      constexpr std::size_t overlap = DistributedNDArrayView::LocalViewType::IndexerType::template getOverlap< dim >();
      if( overlap == 0 ) {
         dim_buffers.reset();
         return;
      }

      using LocalBegins = typename DistributedNDArray::LocalBeginsType;
      using SizesHolder = typename DistributedNDArray::SizesHolderType;
      const LocalBegins& localBegins = array_view.getLocalBegins();
      const SizesHolder& localEnds = array_view.getLocalEnds();

      // offsets for left-send
      dim_buffers.left_send_offsets = localBegins;
      dim_buffers.left_send_offsets.template setSize< dim >( localBegins.template getSize< dim >() - shift );

      // offsets for left-receive
      dim_buffers.left_recv_offsets = localBegins;
      dim_buffers.left_recv_offsets.template setSize< dim >( localBegins.template getSize< dim >() - overlap + shift );

      // offsets for right-send
      dim_buffers.right_send_offsets = localBegins;
      dim_buffers.right_send_offsets.template setSize< dim >( localEnds.template getSize< dim >() - overlap + shift );

      // offsets for right-receive
      dim_buffers.right_recv_offsets = localBegins;
      dim_buffers.right_recv_offsets.template setSize< dim >( localEnds.template getSize< dim >() - shift );
   }

   template< std::size_t dim >
   void
   setNeighbors( int left, int right )
   {
      auto& dim_buffers = buffers.template getDimBuffers< dim >();
      dim_buffers.left_neighbor = left;
      dim_buffers.right_neighbor = right;
   }

   void
   synchronize( DistributedNDArray& array )
   {
      synchronizeAsync( array, AsyncPolicy::synchronous );
   }

   // This method is not thread-safe - only the thread which created and "owns" the
   // instance of this object can call this method.
   // Also note that if (buffered == true), this method must not be called again until
   // the previous asynchronous operation has finished.
   void
   synchronizeAsync( DistributedNDArray& array,
                     AsyncPolicy policy = AsyncPolicy::synchronous,
                     SyncDirection mask = SyncDirection::All )
   {
      // wait for any previous synchronization (multiple objects can share the
      // same synchronizer)
      wait();

      async_start_timer.start();

// GOTCHA: https://devblogs.nvidia.com/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/
#ifdef HAVE_CUDA
      if( std::is_same< typename DistributedNDArray::DeviceType, Devices::Cuda >::value )
         cudaGetDevice( &this->gpu_id );
#endif

      // skip allocation on repeated calls - compare only sizes, not the actual data
      if( array_view.getCommunicator() != array.getCommunicator() || array_view.getSizes() != array.getSizes()
          || array_view.getLocalBegins() != array.getLocalBegins() || array_view.getLocalEnds() != array.getLocalEnds() )
      {
         array_view.bind( array.getView() );
         this->mask = mask;

         // allocate buffers
         Algorithms::staticFor< std::size_t, 0, DistributedNDArray::getDimension() >(
            [ & ]( auto dim )
            {
               allocateHelper< dim >( buffers, array_view );
            } );
      }
      else {
         // only bind to the actual data
         array_view.bind( array.getView() );
         this->mask = mask;
      }

      if( policy == AsyncPolicy::threadpool || policy == AsyncPolicy::async ) {
         // everything offloaded to a separate thread
         auto worker = [ this ]()
         {
// GOTCHA: https://devblogs.nvidia.com/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/
#ifdef HAVE_CUDA
            if( std::is_same< typename DistributedNDArray::DeviceType, Devices::Cuda >::value )
               cudaSetDevice( this->gpu_id );
#endif

            auto requests = this->worker_init();
            MPI::Waitall( requests.data(), requests.size() );
            this->worker_finish();
         };

         if( policy == AsyncPolicy::threadpool )
            async_op = tp.post( worker );
         else
            async_op = std::async( std::launch::async, worker );
      }
      else if( policy == AsyncPolicy::deferred ) {
         // immediate start, deferred synchronization (but still in the same thread)
         auto requests = worker_init();
         auto worker = [ this, requests ]() mutable
         {
            MPI::Waitall( requests.data(), requests.size() );
            this->worker_finish();
         };
         this->async_op = std::async( std::launch::deferred, worker );
      }
      else {
         // synchronous
         auto requests = this->worker_init();
         MPI::Waitall( requests.data(), requests.size() );
         this->worker_finish();
      }

      async_ops_count++;
      async_start_timer.stop();
   }

   void
   wait()
   {
      if( async_op.valid() ) {
         async_wait_timer.start();
         async_op.wait();
         async_wait_timer.stop();
      }
   }

   ~DistributedNDArraySynchronizer()
   {
      if( this->async_op.valid() )
         this->async_op.wait();
   }

   /**
    * \brief Can be used for checking if a synchronization started
    * asynchronously has been finished.
    */
   std::future< void > async_op;

   // attributes for profiling
   Timer async_start_timer, async_wait_timer;
   std::size_t async_ops_count = 0;

protected:
   using DistributedNDArrayView = typename DistributedNDArray::ViewType;
   using Buffers = __ndarray_impl::SynchronizerBuffers< DistributedNDArray >;

   DistributedNDArrayView array_view;
   SyncDirection mask = SyncDirection::All;
   Buffers buffers;

   RequestsVector
   worker_init()
   {
      // fill send buffers
      Algorithms::staticFor< std::size_t, 0, DistributedNDArray::getDimension() >(
         [ & ]( auto dim )
         {
            copyHelper< dim >( buffers, array_view, true, mask );
         } );

      // set default tags from tag_offset
      if( tag_from_left < 0 && tag_to_left < 0 && tag_from_right < 0 && tag_to_right < 0 ) {
         tag_from_left = tag_offset + 1;
         tag_to_left = tag_offset;
         tag_from_right = tag_offset;
         tag_to_right = tag_offset + 1;
      }

      // issue all send and receive async operations
      RequestsVector requests;
      const MPI::Comm& communicator = array_view.getCommunicator();
      Algorithms::staticFor< std::size_t, 0, DistributedNDArray::getDimension() >(
         [ & ]( auto dim )
         {
            sendHelper< dim >(
               buffers, requests, communicator, tag_from_left, tag_to_left, tag_from_right, tag_to_right, mask );
         } );

      return requests;
   }

   void
   worker_finish()
   {
      // copy data from receive buffers
      Algorithms::staticFor< std::size_t, 0, DistributedNDArray::getDimension() >(
         [ & ]( auto dim )
         {
            copyHelper< dim >( buffers, array_view, false, mask );
         } );
   }

   template< std::size_t dim >
   static void
   allocateHelper( Buffers& buffers, const DistributedNDArrayView& array_view )
   {
      auto& dim_buffers = buffers.template getDimBuffers< dim >();

      constexpr std::size_t overlap = DistributedNDArrayView::LocalViewType::IndexerType::template getOverlap< dim >();
      if( overlap == 0 ) {
         dim_buffers.reset();
         return;
      }

      using LocalBegins = typename DistributedNDArray::LocalBeginsType;
      using SizesHolder = typename DistributedNDArray::SizesHolderType;
      const LocalBegins& localBegins = array_view.getLocalBegins();
      const SizesHolder& localEnds = array_view.getLocalEnds();

      SizesHolder bufferSize( localEnds );
      bufferSize.template setSize< dim >( overlap );

      // allocate buffers
      dim_buffers.left_send_buffer.setSize( bufferSize );
      dim_buffers.left_recv_buffer.setSize( bufferSize );
      dim_buffers.right_send_buffer.setSize( bufferSize );
      dim_buffers.right_recv_buffer.setSize( bufferSize );

      // bind views to the buffers
      dim_buffers.left_send_view.bind( dim_buffers.left_send_buffer.getView() );
      dim_buffers.left_recv_view.bind( dim_buffers.left_recv_buffer.getView() );
      dim_buffers.right_send_view.bind( dim_buffers.right_send_buffer.getView() );
      dim_buffers.right_recv_view.bind( dim_buffers.right_recv_buffer.getView() );

      // TODO: check overlap offsets for 2D and 3D distributions (watch out for the corners - maybe use
      // SetSizesSubtractOverlapsHelper?)

      // offsets for left-send
      dim_buffers.left_send_offsets = localBegins;
      dim_buffers.left_send_offsets.template setSize< dim >( localBegins.template getSize< dim >() );

      // offsets for left-receive
      dim_buffers.left_recv_offsets = localBegins;
      dim_buffers.left_recv_offsets.template setSize< dim >( localBegins.template getSize< dim >() - overlap );

      // offsets for right-send
      dim_buffers.right_send_offsets = localBegins;
      dim_buffers.right_send_offsets.template setSize< dim >( localEnds.template getSize< dim >() - overlap );

      // offsets for right-receive
      dim_buffers.right_recv_offsets = localBegins;
      dim_buffers.right_recv_offsets.template setSize< dim >( localEnds.template getSize< dim >() );

      // set default neighbor IDs
      const MPI::Comm& communicator = array_view.getCommunicator();
      const int rank = communicator.rank();
      const int nproc = communicator.size();
      if( dim_buffers.left_neighbor < 0 )
         dim_buffers.left_neighbor = ( rank + nproc - 1 ) % nproc;
      if( dim_buffers.right_neighbor < 0 )
         dim_buffers.right_neighbor = ( rank + 1 ) % nproc;
   }

   template< std::size_t dim >
   static void
   copyHelper( Buffers& buffers, DistributedNDArrayView& array_view, bool to_buffer, SyncDirection mask )
   {
      // skip if there are no overlaps
      constexpr std::size_t overlap = DistributedNDArrayView::LocalViewType::IndexerType::template getOverlap< dim >();
      if( overlap == 0 )
         return;

      auto& dim_buffers = buffers.template getDimBuffers< dim >();

      if( buffered ) {
         // TODO: specify CUDA stream for the copy, otherwise async won't work !!!
         CopyKernel< decltype( dim_buffers.left_send_view ) > copy_kernel;
         copy_kernel.array_view.bind( array_view );
         copy_kernel.to_buffer = to_buffer;

         if( to_buffer ) {
            if( ( mask & SyncDirection::Left ) != SyncDirection::None ) {
               copy_kernel.buffer_view.bind( dim_buffers.left_send_view );
               copy_kernel.array_offsets = dim_buffers.left_send_offsets;
               dim_buffers.left_send_view.forAll( copy_kernel );
            }

            if( ( mask & SyncDirection::Right ) != SyncDirection::None ) {
               copy_kernel.buffer_view.bind( dim_buffers.right_send_view );
               copy_kernel.array_offsets = dim_buffers.right_send_offsets;
               dim_buffers.right_send_view.forAll( copy_kernel );
            }
         }
         else {
            if( ( mask & SyncDirection::Right ) != SyncDirection::None ) {
               copy_kernel.buffer_view.bind( dim_buffers.left_recv_view );
               copy_kernel.array_offsets = dim_buffers.left_recv_offsets;
               dim_buffers.left_recv_view.forAll( copy_kernel );
            }

            if( ( mask & SyncDirection::Left ) != SyncDirection::None ) {
               copy_kernel.buffer_view.bind( dim_buffers.right_recv_view );
               copy_kernel.array_offsets = dim_buffers.right_recv_offsets;
               dim_buffers.right_recv_view.forAll( copy_kernel );
            }
         }
      }
      else {
         // avoid buffering - bind buffer views directly to the array
         dim_buffers.left_send_view.bind( &call_with_offsets( dim_buffers.left_send_offsets, array_view ) );
         dim_buffers.left_recv_view.bind( &call_with_offsets( dim_buffers.left_recv_offsets, array_view ) );
         dim_buffers.right_send_view.bind( &call_with_offsets( dim_buffers.right_send_offsets, array_view ) );
         dim_buffers.right_recv_view.bind( &call_with_offsets( dim_buffers.right_recv_offsets, array_view ) );
      }
   }

   template< std::size_t dim >
   static void
   sendHelper( Buffers& buffers,
               RequestsVector& requests,
               const MPI::Comm& communicator,
               int tag_from_left,
               int tag_to_left,
               int tag_from_right,
               int tag_to_right,
               SyncDirection mask )
   {
      constexpr std::size_t overlap = DistributedNDArrayView::LocalViewType::IndexerType::template getOverlap< dim >();
      if( overlap == 0 )
         return;

      auto& dim_buffers = buffers.template getDimBuffers< dim >();

      if( ( mask & SyncDirection::Left ) != SyncDirection::None ) {
         // negative tags are not valid according to the MPI standard and may be used by
         // applications to skip communication, e.g. over the periodic boundary
         if( tag_to_left >= 0 ) {
            requests.push_back( MPI::Isend( dim_buffers.left_send_view.getData(),
                                            dim_buffers.left_send_view.getStorageSize(),
                                            dim_buffers.left_neighbor,
                                            tag_to_left,
                                            communicator ) );
         }
         if( tag_from_right >= 0 ) {
            requests.push_back( MPI::Irecv( dim_buffers.right_recv_view.getData(),
                                            dim_buffers.right_recv_view.getStorageSize(),
                                            dim_buffers.right_neighbor,
                                            tag_from_right,
                                            communicator ) );
         }
      }
      if( ( mask & SyncDirection::Right ) != SyncDirection::None ) {
         // negative tags are not valid according to the MPI standard and may be used by
         // applications to skip communication, e.g. over the periodic boundary
         if( tag_to_right >= 0 ) {
            requests.push_back( MPI::Isend( dim_buffers.right_send_view.getData(),
                                            dim_buffers.right_send_view.getStorageSize(),
                                            dim_buffers.right_neighbor,
                                            tag_to_right,
                                            communicator ) );
         }
         if( tag_from_left >= 0 ) {
            requests.push_back( MPI::Irecv( dim_buffers.left_recv_view.getData(),
                                            dim_buffers.left_recv_view.getStorageSize(),
                                            dim_buffers.left_neighbor,
                                            tag_from_left,
                                            communicator ) );
         }
      }
   }

#ifdef __NVCC__
public:
#endif
   template< typename BufferView >
   struct CopyKernel
   {
      using ArrayView = typename DistributedNDArray::ViewType;
      using LocalBegins = typename ArrayView::LocalBeginsType;

      BufferView buffer_view;
      ArrayView array_view;
      LocalBegins array_offsets;
      bool to_buffer;

      template< typename... Indices >
      __cuda_callable__
      void
      operator()( Indices... indices )
      {
         if( to_buffer )
            buffer_view( indices... ) = call_with_shifted_indices( array_offsets, array_view, indices... );
         else
            call_with_shifted_indices( array_offsets, array_view, indices... ) = buffer_view( indices... );
      }
   };
};

}  // namespace Containers
}  // namespace TNL
