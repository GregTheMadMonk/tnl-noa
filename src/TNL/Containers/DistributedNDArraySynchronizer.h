/***************************************************************************
                          DistributedNDArraySynchronizer.h  -  description
                             -------------------
    begin                : Mar 30, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <future>

#include <TNL/Containers/ndarray/SynchronizerBuffers.h>

namespace TNL {
namespace Containers {

template< typename DistributedNDArray >
class DistributedNDArraySynchronizer
{
public:
   void synchronize( DistributedNDArray& array )
   {
      auto future = synchronizeAsync( array, std::launch::deferred );
      future.wait();
   }

   // This method is not thread-safe - only the thread which created and "owns" the
   // instance of this object can call this method.
   // Also note that this method must not be called again until the previous
   // asynchronous operation has finished.
   std::shared_future<void> synchronizeAsync( DistributedNDArray& array, std::launch policy = std::launch::async )
   {
      // GOTCHA: https://devblogs.nvidia.com/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/
      #ifdef HAVE_CUDA
      if( std::is_same< typename DistributedNDArray::DeviceType, Devices::Cuda >::value )
         cudaGetDevice(&this->gpu_id);
      #endif

      // NOTE: the allocation cannot be done in the worker, otherwise CUDA would crash
      // skip allocation on repeated calls - compare only sizes, not the actual data
      if( array_view.getCommunicationGroup() != array.getCommunicationGroup() ||
          array_view.getSizes() != array.getSizes() ||
          array_view.getLocalBegins() != array.getLocalBegins() ||
          array_view.getLocalEnds() != array.getLocalEnds() )
      {
         array_view.bind( array.getView() );

         // allocate buffers
         TemplateStaticFor< std::size_t, 0, DistributedNDArray::getDimension(), AllocateHelper >::execHost( buffers, array_view );
      }
      else {
         // only bind to the actual data
         array_view.bind( array.getView() );
      }

      auto worker = [this](){ this->worker(); };
      return std::async( policy, worker );
   }

protected:
   using DistributedNDArrayView = typename DistributedNDArray::ViewType;
   using Communicator = typename DistributedNDArray::CommunicatorType;
   using Buffers = __ndarray_impl::SynchronizerBuffers< DistributedNDArray >;

   DistributedNDArrayView array_view;
   Buffers buffers;
   int gpu_id = 0;

   void worker()
   {
      // GOTCHA: https://devblogs.nvidia.com/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/
      #ifdef HAVE_CUDA
      if( std::is_same< typename DistributedNDArray::DeviceType, Devices::Cuda >::value )
         cudaSetDevice(gpu_id);
      #endif

      // fill send buffers
      TemplateStaticFor< std::size_t, 0, DistributedNDArray::getDimension(), CopyHelper >::execHost( buffers, array_view, true );

      // issue all send and receive async operations
      std::vector< typename Communicator::Request > requests;
      const typename Communicator::CommunicationGroup group = array_view.getCommunicationGroup();
      TemplateStaticFor< std::size_t, 0, DistributedNDArray::getDimension(), SendHelper >::execHost( buffers, requests, group );

      // wait until send is done
      Communicator::WaitAll( requests.data(), requests.size() );

      // copy data from receive buffers
      TemplateStaticFor< std::size_t, 0, DistributedNDArray::getDimension(), CopyHelper >::execHost( buffers, array_view, false );
   }

   template< std::size_t dim >
   struct AllocateHelper
   {
      static void exec( Buffers& buffers, const DistributedNDArrayView& array_view )
      {
         auto& dim_buffers = buffers.template getDimBuffers< dim >();

         constexpr std::size_t overlap = __ndarray_impl::get< dim >( typename DistributedNDArray::OverlapsType{} );
         // TODO
//         constexpr std::size_t overlap = array_view.template getOverlap< dim >();
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

         dim_buffers.left_send_buffer.setSize( bufferSize );
         dim_buffers.left_recv_buffer.setSize( bufferSize );
         dim_buffers.right_send_buffer.setSize( bufferSize );
         dim_buffers.right_recv_buffer.setSize( bufferSize );

         // TODO: check overlap offsets for 2D and 3D distributions (watch out for the corners - maybe use SetSizesSubtractOverlapsHelper?)

         // offsets for left-send
         dim_buffers.left_send_offsets = localBegins;

         // offsets for left-receive
         dim_buffers.left_recv_offsets = localBegins;
         dim_buffers.left_recv_offsets.template setSize< dim >( localBegins.template getSize< dim >() - overlap );

         // offsets for right-send
         dim_buffers.right_send_offsets = localBegins;
         dim_buffers.right_send_offsets.template setSize< dim >( localEnds.template getSize< dim >() - overlap );

         // offsets for right-receive
         dim_buffers.right_recv_offsets = localBegins;
         dim_buffers.right_recv_offsets.template setSize< dim >( localEnds.template getSize< dim >() );

         // FIXME: set proper neighbor IDs !!!
         const typename Communicator::CommunicationGroup group = array_view.getCommunicationGroup();
         const int rank = Communicator::GetRank(group);
         const int nproc = Communicator::GetSize(group);
         dim_buffers.left_neighbor = (rank + nproc - 1) % nproc;
         dim_buffers.right_neighbor = (rank + 1) % nproc;
      }
   };

   template< std::size_t dim >
   struct CopyHelper
   {
      static void exec( Buffers& buffers, DistributedNDArrayView& array_view, bool to_buffer )
      {
         const std::size_t overlap = __ndarray_impl::get< dim >( typename DistributedNDArray::OverlapsType{} );
         if( overlap == 0 )
            return;

         auto& dim_buffers = buffers.template getDimBuffers< dim >();

         // TODO: specify CUDA stream for the copy, otherwise async won't work !!!
         CopyKernel< decltype(dim_buffers.left_send_buffer.getView()) > copy_kernel;
         copy_kernel.array_view.bind( array_view );
         copy_kernel.to_buffer = to_buffer;

         if( to_buffer ) {
            copy_kernel.buffer_view.bind( dim_buffers.left_send_buffer.getView() );
            copy_kernel.array_offsets = dim_buffers.left_send_offsets;
            dim_buffers.left_send_buffer.forAll( copy_kernel );

            copy_kernel.buffer_view.bind( dim_buffers.right_send_buffer.getView() );
            copy_kernel.array_offsets = dim_buffers.right_send_offsets;
            dim_buffers.right_send_buffer.forAll( copy_kernel );
         }
         else {
            copy_kernel.buffer_view.bind( dim_buffers.left_recv_buffer.getView() );
            copy_kernel.array_offsets = dim_buffers.left_recv_offsets;
            dim_buffers.left_recv_buffer.forAll( copy_kernel );

            copy_kernel.buffer_view.bind( dim_buffers.right_recv_buffer.getView() );
            copy_kernel.array_offsets = dim_buffers.right_recv_offsets;
            dim_buffers.right_recv_buffer.forAll( copy_kernel );
         }
      }
   };

   template< std::size_t dim >
   struct SendHelper
   {
      template< typename Requests, typename Group >
      static void exec( Buffers& buffers, Requests& requests, Group group )
      {
         const std::size_t overlap = __ndarray_impl::get< dim >( typename DistributedNDArray::OverlapsType{} );
         if( overlap == 0 )
            return;

         auto& dim_buffers = buffers.template getDimBuffers< dim >();

         requests.push_back( Communicator::ISend( dim_buffers.left_send_buffer.getStorageArray().getData(),
                                                  dim_buffers.left_send_buffer.getStorageSize(),
                                                  dim_buffers.left_neighbor, 0, group ) );
         requests.push_back( Communicator::IRecv( dim_buffers.left_recv_buffer.getStorageArray().getData(),
                                                  dim_buffers.left_recv_buffer.getStorageSize(),
                                                  dim_buffers.left_neighbor, 1, group ) );
         requests.push_back( Communicator::ISend( dim_buffers.right_send_buffer.getStorageArray().getData(),
                                                  dim_buffers.right_send_buffer.getStorageSize(),
                                                  dim_buffers.right_neighbor, 1, group ) );
         requests.push_back( Communicator::IRecv( dim_buffers.right_recv_buffer.getStorageArray().getData(),
                                                  dim_buffers.right_recv_buffer.getStorageSize(),
                                                  dim_buffers.right_neighbor, 0, group ) );
      }
   };

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
      void operator()( Indices... indices )
      {
         if( to_buffer )
            buffer_view( indices... ) = call_with_shifted_indices( array_offsets, array_view, indices... );
         else
            call_with_shifted_indices( array_offsets, array_view, indices... ) = buffer_view( indices... );
      }
   };
};

} // namespace Containers
} // namespace TNL
