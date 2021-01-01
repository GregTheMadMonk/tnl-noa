/***************************************************************************
                          Partitioner.h  -  description
                             -------------------
    begin                : Sep 6, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <vector>

#include "Subrange.h"
#include "ByteArraySynchronizer.h"

#include <TNL/Math.h>

namespace TNL {
namespace Containers {

template< typename Index, typename Communicator = Communicators::MpiCommunicator >
class Partitioner
{
   using CommunicationGroup = typename Communicator::CommunicationGroup;
public:
   using SubrangeType = Subrange< Index >;

   static SubrangeType splitRange( Index globalSize, CommunicationGroup group )
   {
      if( group != Communicator::NullGroup ) {
         const int rank = Communicator::GetRank( group );
         const int partitions = Communicator::GetSize( group );
         const Index begin = TNL::min( globalSize, rank * globalSize / partitions );
         const Index end = TNL::min( globalSize, (rank + 1) * globalSize / partitions );
         return SubrangeType( begin, end );
      }
      else
         return SubrangeType( 0, 0 );
   }

   // Gets the owner of given global index.
   __cuda_callable__
   static int getOwner( Index i, Index globalSize, int partitions )
   {
      int owner = i * partitions / globalSize;
      if( owner < partitions - 1 && i >= getOffset( globalSize, owner + 1, partitions ) )
         owner++;
      TNL_ASSERT_GE( i, getOffset( globalSize, owner, partitions ), "BUG in getOwner" );
      TNL_ASSERT_LT( i, getOffset( globalSize, owner + 1, partitions ), "BUG in getOwner" );
      return owner;
   }

   // Gets the offset of data for given rank.
   __cuda_callable__
   static Index getOffset( Index globalSize, int rank, int partitions )
   {
      return rank * globalSize / partitions;
   }

   // Gets the size of data assigned to given rank.
   __cuda_callable__
   static Index getSizeForRank( Index globalSize, int rank, int partitions )
   {
      const Index begin = min( globalSize, rank * globalSize / partitions );
      const Index end = min( globalSize, (rank + 1) * globalSize / partitions );
      return end - begin;
   }

   template< typename Device >
   class ArraySynchronizer
   : public ByteArraySynchronizer< Device, Index >
   {
      using Base = ByteArraySynchronizer< Device, Index >;

      SubrangeType localRange;
      int overlaps;
      CommunicationGroup group;

   public:
      using ByteArrayView = typename Base::ByteArrayView;
      using RequestsVector = typename Base::RequestsVector;

      ~ArraySynchronizer()
      {
         // wait for pending async operation, otherwise it would crash
         if( this->async_op.valid() )
            this->async_op.wait();
      }

      ArraySynchronizer() = delete;

      ArraySynchronizer( SubrangeType localRange, int overlaps, CommunicationGroup group )
      : localRange(localRange), overlaps(overlaps), group(group)
      {}

      virtual void synchronizeByteArray( ByteArrayView array, int bytesPerValue ) override
      {
         auto requests = synchronizeByteArrayAsyncWorker( array, bytesPerValue );
         Communicator::WaitAll( requests.data(), requests.size() );
      }

      virtual RequestsVector synchronizeByteArrayAsyncWorker( ByteArrayView array, int bytesPerValue ) override
      {
         TNL_ASSERT_EQ( array.getSize(), bytesPerValue * (localRange.getSize() + 2 * overlaps),
                        "unexpected array size" );

         const int rank = Communicator::GetRank( group );
         const int nproc = Communicator::GetSize( group );
         const int left = (rank > 0) ? rank - 1 : nproc - 1;
         const int right = (rank < nproc - 1) ? rank + 1 : 0;

         // buffer for asynchronous communication requests
         std::vector< typename Communicator::Request > requests;

         // issue all async receive operations
         requests.push_back( Communicator::IRecv(
                  array.getData() + bytesPerValue * localRange.getSize(),
                  bytesPerValue * overlaps,
                  left, 0, group ) );
         requests.push_back( Communicator::IRecv(
                  array.getData() + bytesPerValue * (localRange.getSize() + overlaps),
                  bytesPerValue * overlaps,
                  right, 0, group ) );

         // issue all async send operations
         requests.push_back( Communicator::ISend(
                  array.getData(),
                  bytesPerValue * overlaps,
                  left, 0, group ) );
         requests.push_back( Communicator::ISend(
                  array.getData() + bytesPerValue * (localRange.getSize() - overlaps),
                  bytesPerValue * overlaps,
                  right, 0, group ) );

         return requests;
      }
   };
};

} // namespace Containers
} // namespace TNL
