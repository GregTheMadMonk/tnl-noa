/***************************************************************************
                          DistributedVectorTest.h  -  description
                             -------------------
    begin                : Sep 6, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#ifdef HAVE_GTEST
#include <limits>

#include <gtest/gtest.h>

#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Communicators/NoDistrCommunicator.h>
#include <TNL/Containers/DistributedVector.h>
#include <TNL/Containers/DistributedVectorView.h>
#include <TNL/Containers/Partitioner.h>

#define DISTRIBUTED_VECTOR
#include "VectorSequenceSetupFunctions.h"

using namespace TNL;
using namespace TNL::Containers;

/*
 * Light check of DistributedVector.
 *
 * - Number of processes is not limited.
 * - Global size is hardcoded as 97 to force non-uniform distribution.
 * - Communication group is hardcoded as AllGroup -- it may be changed as needed.
 */
template< typename DistributedVector >
class DistributedVectorTest
: public ::testing::Test
{
protected:
   using RealType = typename DistributedVector::RealType;
   using DeviceType = typename DistributedVector::DeviceType;
   using CommunicatorType = typename DistributedVector::CommunicatorType;
   using IndexType = typename DistributedVector::IndexType;
   using DistributedVectorType = DistributedVector;
   using VectorViewType = typename DistributedVectorType::LocalViewType;
   using DistributedVectorView = Containers::DistributedVectorView< RealType, DeviceType, IndexType, CommunicatorType >;

   const int globalSize = 97;  // prime number to force non-uniform distribution

   const typename CommunicatorType::CommunicationGroup group = CommunicatorType::AllGroup;

   DistributedVectorType x, y, z;

   DistributedVectorView x_view, y_view, z_view;

   const int rank = CommunicatorType::GetRank(group);
   const int nproc = CommunicatorType::GetSize(group);

   DistributedVectorTest()
   {
      using LocalRangeType = typename DistributedVector::LocalRangeType;
      const LocalRangeType localRange = Partitioner< IndexType, CommunicatorType >::splitRange( globalSize, group );
      x.setDistribution( localRange, globalSize, group );
      y.setDistribution( localRange, globalSize, group );
      z.setDistribution( localRange, globalSize, group );

      x_view.bind( x );
      y_view.bind( y );
      z_view.bind( z );

      setConstantSequence( x, 1 );
      setLinearSequence( y );
      setNegativeLinearSequence( z );
   }
};

// types for which DistributedVectorTest is instantiated
using DistributedVectorTypes = ::testing::Types<
   DistributedVector< double, Devices::Host, int, Communicators::MpiCommunicator >,
   DistributedVector< double, Devices::Host, int, Communicators::NoDistrCommunicator >
#ifdef HAVE_CUDA
   ,
   DistributedVector< double, Devices::Cuda, int, Communicators::MpiCommunicator >,
   DistributedVector< double, Devices::Cuda, int, Communicators::NoDistrCommunicator >
#endif
>;

TYPED_TEST_SUITE( DistributedVectorTest, DistributedVectorTypes );

TYPED_TEST( DistributedVectorTest, addVector )
{
   this->x.addVector( this->y, 3.0, 1.0 );
   for( int i = 0; i < this->y.getLocalView().getSize(); i++ ) {
      const auto gi = this->y.getLocalRange().getGlobalIndex( i );
      EXPECT_EQ( this->x.getElement( gi ), 1 + 3 * gi );
   }

   this->y += this->z;
   for( int i = 0; i < this->y.getLocalView().getSize(); i++ ) {
      const auto gi = this->y.getLocalRange().getGlobalIndex( i );
      EXPECT_EQ( this->y.getElement( gi ), 0 );
   }

   setConstantSequence( this->x, 1 );
   setLinearSequence( this->y );

   this->x_view.addVector( this->y_view, 3.0, 1.0 );
   for( int i = 0; i < this->y_view.getLocalView().getSize(); i++ ) {
      const auto gi = this->y_view.getLocalRange().getGlobalIndex( i );
      EXPECT_EQ( this->x_view.getElement( gi ), 1 + 3 * gi );
   }

   this->y_view += this->z_view;
   for( int i = 0; i < this->y_view.getLocalView().getSize(); i++ ) {
      const auto gi = this->y_view.getLocalRange().getGlobalIndex( i );
      EXPECT_EQ( this->y_view.getElement( gi ), 0 );
   }
}

TYPED_TEST( DistributedVectorTest, addVectors )
{
   this->x.addVectors( this->y, 3.0, this->z, 1.0, 2.0 );
   for( int i = 0; i < this->y.getLocalView().getSize(); i++ ) {
      const auto gi = this->y.getLocalRange().getGlobalIndex( i );
      EXPECT_EQ( this->x.getElement( gi ), 2 + 3 * gi - gi );
   }

   setConstantSequence( this->x, 1 );

   this->x_view.addVectors( this->y_view, 3.0, this->z_view, 1.0, 2.0 );
   for( int i = 0; i < this->y_view.getLocalView().getSize(); i++ ) {
      const auto gi = this->y_view.getLocalRange().getGlobalIndex( i );
      EXPECT_EQ( this->x_view.getElement( gi ), 2 + 3 * gi - gi );
   }
}

// TODO: distributed prefix sum

#endif  // HAVE_GTEST

#include "../main_mpi.h"
