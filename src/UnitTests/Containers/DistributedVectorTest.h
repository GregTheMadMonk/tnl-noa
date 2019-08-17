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
#include "VectorHelperFunctions.h"

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

   const typename CommunicatorType::CommunicationGroup group = CommunicatorType::AllGroup;

   DistributedVectorType v;
   DistributedVectorView v_view;
   typename DistributedVectorType::HostType v_host;

   const int rank = CommunicatorType::GetRank(group);
   const int nproc = CommunicatorType::GetSize(group);

   // should be small enough to have fast tests, but large enough to test
   // prefix-sum with multiple CUDA grids
   const int globalSize = 10000 * nproc;

   DistributedVectorTest()
   {
      using LocalRangeType = typename DistributedVector::LocalRangeType;
      const LocalRangeType localRange = Partitioner< IndexType, CommunicatorType >::splitRange( globalSize, group );
      v.setDistribution( localRange, globalSize, group );
      v_view.bind( v );
      setConstantSequence( v, 1 );
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

#if 1
TYPED_TEST( DistributedVectorTest, prefixSum )
{
   using RealType = typename TestFixture::DistributedVectorType::RealType;
   using DeviceType = typename TestFixture::DistributedVectorType::DeviceType;
   using IndexType = typename TestFixture::DistributedVectorType::IndexType;

   auto& v = this->v;
   auto& v_view = this->v_view;
   auto& v_host = this->v_host;
   const auto localRange = v.getLocalRange();

   // FIXME: tests should work in all cases
   if( std::is_same< RealType, float >::value )
      return;

   setConstantSequence( v, 0 );
   v_host = -1;
   v.prefixSum();
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host = -1;
   v.prefixSum();
   v_host = v_view;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], i + 1 ) << "i = " << i;

   setLinearSequence( v );
   v_host = -1;
   v.prefixSum();
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

   // test views
   setConstantSequence( v, 0 );
   v_host = -1;
   v_view.prefixSum();
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host = -1;
   v_view.prefixSum();
   v_host = v_view;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], i + 1 ) << "i = " << i;

   setLinearSequence( v );
   v_host = -1;
   v_view.prefixSum();
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

   ////
   // With CUDA, perform tests with multiple CUDA grids.
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      Algorithms::CudaPrefixSumKernelLauncher< Algorithms::PrefixSumType::Inclusive, RealType, IndexType >::maxGridSize() = 3;

      setConstantSequence( v, 0 );
      v_host = -1;
      v.prefixSum();
      EXPECT_GT( ( Algorithms::CudaPrefixSumKernelLauncher< Algorithms::PrefixSumType::Inclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], 0 );

      setConstantSequence( v, 1 );
      v_host = -1;
      v.prefixSum();
      EXPECT_GT( ( Algorithms::CudaPrefixSumKernelLauncher< Algorithms::PrefixSumType::Inclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v_view;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], i + 1 );

      setLinearSequence( v );
      v_host = -1;
      v.prefixSum();
      EXPECT_GT( ( Algorithms::CudaPrefixSumKernelLauncher< Algorithms::PrefixSumType::Inclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

      // test views
      setConstantSequence( v, 0 );
      v_host = -1;
      v_view.prefixSum();
      EXPECT_GT( ( Algorithms::CudaPrefixSumKernelLauncher< Algorithms::PrefixSumType::Inclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], 0 );

      setConstantSequence( v, 1 );
      v_host = -1;
      v_view.prefixSum();
      EXPECT_GT( ( Algorithms::CudaPrefixSumKernelLauncher< Algorithms::PrefixSumType::Inclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v_view;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], i + 1 );

      setLinearSequence( v );
      v_host = -1;
      v_view.prefixSum();
      EXPECT_GT( ( Algorithms::CudaPrefixSumKernelLauncher< Algorithms::PrefixSumType::Inclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

      Algorithms::CudaPrefixSumKernelLauncher< Algorithms::PrefixSumType::Inclusive, RealType, IndexType >::resetMaxGridSize();
#endif
   }
}
#endif

#if 1
TYPED_TEST( DistributedVectorTest, exclusivePrefixSum )
{
   using RealType = typename TestFixture::DistributedVectorType::RealType;
   using DeviceType = typename TestFixture::DistributedVectorType::DeviceType;
   using IndexType = typename TestFixture::DistributedVectorType::IndexType;

   auto& v = this->v;
   auto& v_view = this->v_view;
   auto& v_host = this->v_host;
   const auto localRange = v.getLocalRange();

   // FIXME: tests should work in all cases
   if( std::is_same< RealType, float >::value )
      return;

   setConstantSequence( v, 0 );
   v_host = -1;
   v.template prefixSum< Algorithms::PrefixSumType::Exclusive >();
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host = -1;
   v.template prefixSum< Algorithms::PrefixSumType::Exclusive >();
   v_host = v_view;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], i ) << "i = " << i;

   setLinearSequence( v );
   v_host = -1;
   v.template prefixSum< Algorithms::PrefixSumType::Exclusive >();
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

   // test views
   setConstantSequence( v, 0 );
   v_host = -1;
   v_view.template prefixSum< Algorithms::PrefixSumType::Exclusive >();
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host = -1;
   v_view.template prefixSum< Algorithms::PrefixSumType::Exclusive >();
   v_host = v_view;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], i ) << "i = " << i;

   setLinearSequence( v );
   v_host = -1;
   v_view.template prefixSum< Algorithms::PrefixSumType::Exclusive >();
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

   ////
   // With CUDA, perform tests with multiple CUDA grids.
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      Algorithms::CudaPrefixSumKernelLauncher< Algorithms::PrefixSumType::Exclusive, RealType, IndexType >::maxGridSize() = 3;

      setConstantSequence( v, 0 );
      v_host = -1;
      v.template prefixSum< Algorithms::PrefixSumType::Exclusive >();
      EXPECT_GT( ( Algorithms::CudaPrefixSumKernelLauncher< Algorithms::PrefixSumType::Exclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], 0 );

      setConstantSequence( v, 1 );
      v_host = -1;
      v.template prefixSum< Algorithms::PrefixSumType::Exclusive >();
      EXPECT_GT( ( Algorithms::CudaPrefixSumKernelLauncher< Algorithms::PrefixSumType::Exclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v_view;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], i );

      setLinearSequence( v );
      v_host = -1;
      v.template prefixSum< Algorithms::PrefixSumType::Exclusive >();
      EXPECT_GT( ( Algorithms::CudaPrefixSumKernelLauncher< Algorithms::PrefixSumType::Exclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

      // test views
      setConstantSequence( v, 0 );
      v_host = -1;
      v_view.template prefixSum< Algorithms::PrefixSumType::Exclusive >();
      EXPECT_GT( ( Algorithms::CudaPrefixSumKernelLauncher< Algorithms::PrefixSumType::Exclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], 0 );

      setConstantSequence( v, 1 );
      v_host = -1;
      v_view.template prefixSum< Algorithms::PrefixSumType::Exclusive >();
      EXPECT_GT( ( Algorithms::CudaPrefixSumKernelLauncher< Algorithms::PrefixSumType::Exclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v_view;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], i );

      setLinearSequence( v );
      v_host = -1;
      v_view.template prefixSum< Algorithms::PrefixSumType::Exclusive >();
      EXPECT_GT( ( Algorithms::CudaPrefixSumKernelLauncher< Algorithms::PrefixSumType::Exclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

      Algorithms::CudaPrefixSumKernelLauncher< Algorithms::PrefixSumType::Exclusive, RealType, IndexType >::resetMaxGridSize();
#endif
   }
}
#endif

#endif  // HAVE_GTEST

#include "../main_mpi.h"
