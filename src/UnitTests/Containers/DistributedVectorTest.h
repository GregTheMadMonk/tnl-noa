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

#include <TNL/Containers/DistributedVector.h>
#include <TNL/Containers/DistributedVectorView.h>
#include <TNL/Containers/Partitioner.h>

#define DISTRIBUTED_VECTOR
#include "VectorHelperFunctions.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::MPI;

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
   using IndexType = typename DistributedVector::IndexType;
   using DistributedVectorType = DistributedVector;
   using VectorViewType = typename DistributedVectorType::LocalViewType;
   using DistributedVectorView = Containers::DistributedVectorView< RealType, DeviceType, IndexType >;
   using HostDistributedVectorType = typename DistributedVectorType::template Self< RealType, Devices::Sequential >;

   const MPI_Comm group = AllGroup();

   DistributedVectorType v;
   DistributedVectorView v_view;
   HostDistributedVectorType v_host;

   const int rank = GetRank(group);
   const int nproc = GetSize(group);

   // should be small enough to have fast tests, but large enough to test
   // scan with multiple CUDA grids
   const int globalSize = 10000 * nproc;

   // some arbitrary value (but must be 0 if not distributed)
   const int ghosts = (nproc > 1) ? 4 : 0;

   DistributedVectorTest()
   {
      using LocalRangeType = typename DistributedVector::LocalRangeType;
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

// types for which DistributedVectorTest is instantiated
using DistributedVectorTypes = ::testing::Types<
   DistributedVector< double, Devices::Host, int >
#ifdef HAVE_CUDA
   ,
   DistributedVector< double, Devices::Cuda, int >
#endif
>;

TYPED_TEST_SUITE( DistributedVectorTest, DistributedVectorTypes );

// TODO: test that horizontal operations are computed for ghost values without synchronization

TYPED_TEST( DistributedVectorTest, scan )
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
   v.scan();
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host = -1;
   v.scan();
   v_host = v_view;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], i + 1 ) << "i = " << i;

   setLinearSequence( v );
   v_host = -1;
   v.scan();
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

   // test views
   setConstantSequence( v, 0 );
   v_host = -1;
   v_view.scan();
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host = -1;
   v_view.scan();
   v_host = v_view;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], i + 1 ) << "i = " << i;

   setLinearSequence( v );
   v_host = -1;
   v_view.scan();
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

   ////
   // With CUDA, perform tests with multiple CUDA grids.
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, RealType, IndexType >::maxGridSize() = 3;

      setConstantSequence( v, 0 );
      v_host = -1;
      v.scan();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], 0 );

      setConstantSequence( v, 1 );
      v_host = -1;
      v.scan();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v_view;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], i + 1 );

      setLinearSequence( v );
      v_host = -1;
      v.scan();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

      // test views
      setConstantSequence( v, 0 );
      v_host = -1;
      v_view.scan();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], 0 );

      setConstantSequence( v, 1 );
      v_host = -1;
      v_view.scan();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v_view;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], i + 1 );

      setLinearSequence( v );
      v_host = -1;
      v_view.scan();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

      Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, RealType, IndexType >::resetMaxGridSize();
#endif
   }
}

TYPED_TEST( DistributedVectorTest, exclusiveScan )
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
   v.template scan< Algorithms::ScanType::Exclusive >();
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host = -1;
   v.template scan< Algorithms::ScanType::Exclusive >();
   v_host = v_view;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], i ) << "i = " << i;

   setLinearSequence( v );
   v_host = -1;
   v.template scan< Algorithms::ScanType::Exclusive >();
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

   // test views
   setConstantSequence( v, 0 );
   v_host = -1;
   v_view.template scan< Algorithms::ScanType::Exclusive >();
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host = -1;
   v_view.template scan< Algorithms::ScanType::Exclusive >();
   v_host = v_view;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], i ) << "i = " << i;

   setLinearSequence( v );
   v_host = -1;
   v_view.template scan< Algorithms::ScanType::Exclusive >();
   v_host = v;
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
      EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

   ////
   // With CUDA, perform tests with multiple CUDA grids.
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, RealType, IndexType >::maxGridSize() = 3;

      setConstantSequence( v, 0 );
      v_host = -1;
      v.template scan< Algorithms::ScanType::Exclusive >();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], 0 );

      setConstantSequence( v, 1 );
      v_host = -1;
      v.template scan< Algorithms::ScanType::Exclusive >();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v_view;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], i );

      setLinearSequence( v );
      v_host = -1;
      v.template scan< Algorithms::ScanType::Exclusive >();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

      // test views
      setConstantSequence( v, 0 );
      v_host = -1;
      v_view.template scan< Algorithms::ScanType::Exclusive >();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], 0 );

      setConstantSequence( v, 1 );
      v_host = -1;
      v_view.template scan< Algorithms::ScanType::Exclusive >();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v_view;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], i );

      setLinearSequence( v );
      v_host = -1;
      v_view.template scan< Algorithms::ScanType::Exclusive >();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
         EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

      Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, RealType, IndexType >::resetMaxGridSize();
#endif
   }
}

#endif  // HAVE_GTEST

#include "../main_mpi.h"
