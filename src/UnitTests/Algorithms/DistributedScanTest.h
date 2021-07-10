#pragma once

#ifdef HAVE_GTEST
#include <limits>

#include <gtest/gtest.h>

#include <TNL/Containers/DistributedArray.h>
#include <TNL/Containers/DistributedArrayView.h>
#include <TNL/Containers/Partitioner.h>
#include <TNL/Algorithms/DistributedScan.h>

#define DISTRIBUTED_VECTOR
#include "../Containers/VectorHelperFunctions.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;
using namespace TNL::MPI;

/*
 * Light check of DistributedArray.
 *
 * - Number of processes is not limited.
 * - Global size is hardcoded as 97 to force non-uniform distribution.
 * - Communication group is hardcoded as AllGroup -- it may be changed as needed.
 */
template< typename DistributedArray >
class DistributedScanTest
: public ::testing::Test
{
protected:
   using ValueType = typename DistributedArray::ValueType;
   using DeviceType = typename DistributedArray::DeviceType;
   using IndexType = typename DistributedArray::IndexType;
   using DistributedArrayType = DistributedArray;
   using VectorViewType = typename DistributedArrayType::LocalViewType;
   using DistributedArrayView = Containers::DistributedArrayView< ValueType, DeviceType, IndexType >;
   using HostDistributedArrayType = typename DistributedArrayType::template Self< ValueType, Devices::Sequential >;

   const MPI_Comm group = AllGroup();

   DistributedArrayType v;
   DistributedArrayView v_view;
   HostDistributedArrayType v_host;

   const int rank = GetRank(group);
   const int nproc = GetSize(group);

   // should be small enough to have fast tests, but large enough to test
   // scan with multiple CUDA grids
   const int globalSize = 10000 * nproc;

   // some arbitrary value (but must be 0 if not distributed)
   const int ghosts = (nproc > 1) ? 4 : 0;

   DistributedScanTest()
   {
      using LocalRangeType = typename DistributedArray::LocalRangeType;
      const LocalRangeType localRange = Partitioner< IndexType >::splitRange( globalSize, group );
      v.setDistribution( localRange, ghosts, globalSize, group );

      using Synchronizer = typename Partitioner< IndexType >::template ArraySynchronizer< DeviceType >;
      using HostSynchronizer = typename Partitioner< IndexType >::template ArraySynchronizer< Devices::Sequential >;
      v.setSynchronizer( std::make_shared<Synchronizer>( localRange, ghosts / 2, group ) );
      v_view.setSynchronizer( v.getSynchronizer() );
      v_host.setSynchronizer( std::make_shared<HostSynchronizer>( localRange, ghosts / 2, group ) );

      v_view.bind( v );
      setConstantSequence( v, 1 );
   }
};

// types for which DistributedScanTest is instantiated
using DistributedArrayTypes = ::testing::Types<
   DistributedArray< double, Devices::Sequential, int >,
   DistributedArray< double, Devices::Host, int >
#ifdef HAVE_CUDA
   ,
   DistributedArray< double, Devices::Cuda, int >
#endif
>;

TYPED_TEST_SUITE( DistributedScanTest, DistributedArrayTypes );

// TODO: test that horizontal operations are computed for ghost values without synchronization

TYPED_TEST( DistributedScanTest, inclusiveScan )
{
   using ValueType = typename TestFixture::DistributedArrayType::ValueType;
   using DeviceType = typename TestFixture::DistributedArrayType::DeviceType;
   using IndexType = typename TestFixture::DistributedArrayType::IndexType;

   auto& v = this->v;
   auto& v_view = this->v_view;
   auto& v_host = this->v_host;
   const auto localRange = v.getLocalRange();

   setConstantSequence( v, 0 );
   v_host.setValue( -1 );
   DistributedScan< ScanType::Inclusive >::perform( v, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host.setValue( -1 );
   DistributedScan< ScanType::Inclusive >::perform( v, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   v_host = v_view;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], i + 1 ) << "i = " << i;

   setLinearSequence( v );
   v_host.setValue( -1 );
   DistributedScan< ScanType::Inclusive >::perform( v, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

   // test views
   setConstantSequence( v, 0 );
   v_host.setValue( -1 );
   DistributedScan< ScanType::Inclusive >::perform( v_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host.setValue( -1 );
   DistributedScan< ScanType::Inclusive >::perform( v_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   v_host = v_view;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], i + 1 ) << "i = " << i;

   setLinearSequence( v );
   v_host.setValue( -1 );
   DistributedScan< ScanType::Inclusive >::perform( v_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

   ////
   // With CUDA, perform tests with multiple CUDA grids.
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, ValueType, IndexType >::maxGridSize() = 3;

      setConstantSequence( v, 0 );
      v_host.setValue( -1 );
      DistributedScan< ScanType::Inclusive >::perform( v, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, ValueType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], 0 );

      setConstantSequence( v, 1 );
      v_host.setValue( -1 );
      DistributedScan< ScanType::Inclusive >::perform( v, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, ValueType, IndexType >::gridsCount() ), 1  );
      v_host = v_view;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], i + 1 );

      setLinearSequence( v );
      v_host.setValue( -1 );
      DistributedScan< ScanType::Inclusive >::perform( v, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, ValueType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

      // test views
      setConstantSequence( v, 0 );
      v_host.setValue( -1 );
      DistributedScan< ScanType::Inclusive >::perform( v_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, ValueType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], 0 );

      setConstantSequence( v, 1 );
      v_host.setValue( -1 );
      DistributedScan< ScanType::Inclusive >::perform( v_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, ValueType, IndexType >::gridsCount() ), 1  );
      v_host = v_view;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], i + 1 );

      setLinearSequence( v );
      v_host.setValue( -1 );
      DistributedScan< ScanType::Inclusive >::perform( v_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, ValueType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

      Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, ValueType, IndexType >::resetMaxGridSize();
#endif
   }
}

TYPED_TEST( DistributedScanTest, exclusiveScan )
{
   using ValueType = typename TestFixture::DistributedArrayType::ValueType;
   using DeviceType = typename TestFixture::DistributedArrayType::DeviceType;
   using IndexType = typename TestFixture::DistributedArrayType::IndexType;

   auto& v = this->v;
   auto& v_view = this->v_view;
   auto& v_host = this->v_host;
   const auto localRange = v.getLocalRange();

   // FIXME: tests should work in all cases
   if( std::is_same< ValueType, float >::value )
      return;

   setConstantSequence( v, 0 );
   v_host.setValue( -1 );
   DistributedScan< ScanType::Exclusive >::perform( v, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host.setValue( -1 );
   DistributedScan< ScanType::Exclusive >::perform( v, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   v_host = v_view;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], i ) << "i = " << i;

   setLinearSequence( v );
   v_host.setValue( -1 );
   DistributedScan< ScanType::Exclusive >::perform( v, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

   // test views
   setConstantSequence( v, 0 );
   v_host.setValue( -1 );
   DistributedScan< ScanType::Exclusive >::perform( v_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host.setValue( -1 );
   DistributedScan< ScanType::Exclusive >::perform( v_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   v_host = v_view;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], i ) << "i = " << i;

   setLinearSequence( v );
   v_host.setValue( -1 );
   DistributedScan< ScanType::Exclusive >::perform( v_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

   ////
   // With CUDA, perform tests with multiple CUDA grids.
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, ValueType, IndexType >::maxGridSize() = 3;

      setConstantSequence( v, 0 );
      v_host.setValue( -1 );
      DistributedScan< ScanType::Exclusive >::perform( v, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, ValueType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], 0 );

      setConstantSequence( v, 1 );
      v_host.setValue( -1 );
      DistributedScan< ScanType::Exclusive >::perform( v, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, ValueType, IndexType >::gridsCount() ), 1  );
      v_host = v_view;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], i );

      setLinearSequence( v );
      v_host.setValue( -1 );
      DistributedScan< ScanType::Exclusive >::perform( v, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, ValueType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

      // test views
      setConstantSequence( v, 0 );
      v_host.setValue( -1 );
      DistributedScan< ScanType::Exclusive >::perform( v_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, ValueType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], 0 );

      setConstantSequence( v, 1 );
      v_host.setValue( -1 );
      DistributedScan< ScanType::Exclusive >::perform( v_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, ValueType, IndexType >::gridsCount() ), 1  );
      v_host = v_view;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], i );

      setLinearSequence( v );
      v_host.setValue( -1 );
      DistributedScan< ScanType::Exclusive >::perform( v_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, ValueType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

      Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, ValueType, IndexType >::resetMaxGridSize();
#endif
   }
}

#endif  // HAVE_GTEST

#include "../main_mpi.h"
