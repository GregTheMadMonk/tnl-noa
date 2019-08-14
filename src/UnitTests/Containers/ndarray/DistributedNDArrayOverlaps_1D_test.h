/***************************************************************************
                          DistributedNDArrayOverlaps_1D_test.h  -  description
                             -------------------
    begin                : Dec 27, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Communicators/NoDistrCommunicator.h>
#include <TNL/Communicators/ScopedInitializer.h>
#include <TNL/Containers/DistributedNDArray.h>
#include <TNL/Containers/DistributedNDArrayView.h>
#include <TNL/Containers/DistributedNDArraySynchronizer.h>
#include <TNL/Containers/ArrayView.h>
#include <TNL/Containers/Partitioner.h>

using namespace TNL;
using namespace TNL::Containers;

/*
 * Light check of DistributedNDArray.
 *
 * - Number of processes is not limited.
 * - Global size is hardcoded as 97 to force non-uniform distribution.
 * - Communication group is hardcoded as AllGroup -- it may be changed as needed.
 */
template< typename DistributedNDArray >
class DistributedNDArrayOverlaps_1D_test
: public ::testing::Test
{
protected:
   using ValueType = typename DistributedNDArray::ValueType;
   using DeviceType = typename DistributedNDArray::DeviceType;
   using CommunicatorType = typename DistributedNDArray::CommunicatorType;
   using IndexType = typename DistributedNDArray::IndexType;
   using DistributedNDArrayType = DistributedNDArray;

   // TODO: use ndarray
   using LocalArrayType = Array< ValueType, DeviceType, IndexType >;
   using LocalArrayViewType = ArrayView< ValueType, DeviceType, IndexType >;

   const int globalSize = 97;  // prime number to force non-uniform distribution
   const int overlaps = __ndarray_impl::get< 0 >( typename DistributedNDArray::OverlapsType{} );

   const typename CommunicatorType::CommunicationGroup group = CommunicatorType::AllGroup;

   DistributedNDArrayType distributedNDArray;

   const int rank = CommunicatorType::GetRank(group);
   const int nproc = CommunicatorType::GetSize(group);

   DistributedNDArrayOverlaps_1D_test()
   {
      using LocalRangeType = typename DistributedNDArray::LocalRangeType;
      const LocalRangeType localRange = Partitioner< IndexType, CommunicatorType >::splitRange( globalSize, group );
      distributedNDArray.setSizes( globalSize );
      distributedNDArray.template setDistribution< 0 >( localRange.getBegin(), localRange.getEnd(), group );
      distributedNDArray.allocate();

      EXPECT_EQ( distributedNDArray.template getLocalRange< 0 >(), localRange );
      EXPECT_EQ( distributedNDArray.getCommunicationGroup(), group );
   }
};

// types for which DistributedNDArrayOverlaps_1D_test is instantiated
using DistributedNDArrayTypes = ::testing::Types<
   DistributedNDArray< NDArray< double,
                                SizesHolder< int, 0 >,
                                std::index_sequence< 0 >,
                                Devices::Host >,
                       Communicators::MpiCommunicator,
                       std::index_sequence< 2 > >
// TODO: does it make sense for NoDistrCommunicator?
//   DistributedNDArray< NDArray< double,
//                                SizesHolder< int, 0 >,
//                                std::index_sequence< 0 >,
//                                Devices::Host >,
//                       Communicators::NoDistrCommunicator,
//                       std::index_sequence< 2 > >
#ifdef HAVE_CUDA
   ,
   DistributedNDArray< NDArray< double,
                                SizesHolder< int, 0 >,
                                std::index_sequence< 0 >,
                                Devices::Cuda >,
                       Communicators::MpiCommunicator,
                       std::index_sequence< 2 > >
// TODO: does it make sense for NoDistrCommunicator?
//   DistributedNDArray< NDArray< double,
//                                SizesHolder< int, 0 >,
//                                std::index_sequence< 0 >,
//                                Devices::Cuda >,
//                       Communicators::NoDistrCommunicator,
//                       std::index_sequence< 2 > >
#endif
>;

TYPED_TEST_SUITE( DistributedNDArrayOverlaps_1D_test, DistributedNDArrayTypes );

TYPED_TEST( DistributedNDArrayOverlaps_1D_test, checkSumOfLocalSizes )
{
   using CommunicatorType = typename TestFixture::CommunicatorType;

   const auto localRange = this->distributedNDArray.template getLocalRange< 0 >();
   const int localSize = localRange.getEnd() - localRange.getBegin();
   int sumOfLocalSizes = 0;
   CommunicatorType::Allreduce( &localSize, &sumOfLocalSizes, 1, MPI_SUM, this->group );
   EXPECT_EQ( sumOfLocalSizes, this->globalSize );
   EXPECT_EQ( this->distributedNDArray.template getSize< 0 >(), this->globalSize );

   EXPECT_EQ( this->distributedNDArray.getLocalStorageSize(), 2 * this->overlaps + localSize );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_forLocalInternal( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const int overlaps = __ndarray_impl::get< 0 >( typename DistributedArray::OverlapsType{} );
   const auto localRange = a.template getLocalRange< 0 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      a_view( i ) += 1;
   };

   a.setValue( 0 );
   a.forLocalInternal( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getBegin() + overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
   for( int gi = localRange.getBegin() + overlaps; gi < localRange.getEnd() - overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
   for( int gi = localRange.getEnd() - overlaps; gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;

   a.setValue( 0 );
   a_view.forLocalInternal( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getBegin() + overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
   for( int gi = localRange.getBegin() + overlaps; gi < localRange.getEnd() - overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
   for( int gi = localRange.getEnd() - overlaps; gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
}

TYPED_TEST( DistributedNDArrayOverlaps_1D_test, forLocalInternal )
{
   test_helper_forLocalInternal( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_forLocalBoundary( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const int overlaps = __ndarray_impl::get< 0 >( typename DistributedArray::OverlapsType{} );
   const auto localRange = a.template getLocalRange< 0 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      a_view( i ) += 1;
   };

   a.setValue( 0 );
   a.forLocalBoundary( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getBegin() + overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
   for( int gi = localRange.getBegin() + overlaps; gi < localRange.getEnd() - overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
   for( int gi = localRange.getEnd() - overlaps; gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;

   a.setValue( 0 );
   a_view.forLocalBoundary( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getBegin() + overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
   for( int gi = localRange.getBegin() + overlaps; gi < localRange.getEnd() - overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
   for( int gi = localRange.getEnd() - overlaps; gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
}

TYPED_TEST( DistributedNDArrayOverlaps_1D_test, forLocalBoundary )
{
   test_helper_forLocalBoundary( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_forOverlaps( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const int overlaps = __ndarray_impl::get< 0 >( typename DistributedArray::OverlapsType{} );
   const auto localRange = a.template getLocalRange< 0 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      a_view( i ) += 1;
   };

   a.setValue( 0 );
   a.forOverlaps( setter );

   for( int gi = localRange.getBegin() - overlaps; gi < localRange.getBegin(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
   for( int gi = localRange.getEnd(); gi < localRange.getEnd() + overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;

   a.setValue( 0 );
   a_view.forOverlaps( setter );

   for( int gi = localRange.getBegin() - overlaps; gi < localRange.getBegin(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
   for( int gi = localRange.getEnd(); gi < localRange.getEnd() + overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
}

TYPED_TEST( DistributedNDArrayOverlaps_1D_test, forOverlaps )
{
   test_helper_forOverlaps( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_synchronize( DistributedArray& a, const int rank, const int nproc )
{
   using IndexType = typename DistributedArray::IndexType;

   const int overlaps = __ndarray_impl::get< 0 >( typename DistributedArray::OverlapsType{} );
   const auto localRange = a.template getLocalRange< 0 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      a_view( i ) = i;
   };

   a.setValue( -1 );
   a.forAll( setter );
   DistributedNDArraySynchronizer< DistributedArray > s1;
   s1.synchronize( a );

   for( int gi = localRange.getBegin() - overlaps; gi < localRange.getBegin(); gi++ )
      EXPECT_EQ( a.getElement( gi ), gi + ((rank == 0) ? 97 : 0) );
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), gi );
   for( int gi = localRange.getEnd(); gi < localRange.getEnd() + overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), gi - ((rank == nproc-1) ? 97 : 0) );

   a.setValue( -1 );
   a_view.forAll( setter );
   DistributedNDArraySynchronizer< decltype(a_view) > s2;
   s2.synchronize( a_view );

   for( int gi = localRange.getBegin() - overlaps; gi < localRange.getBegin(); gi++ )
      EXPECT_EQ( a.getElement( gi ), gi + ((rank == 0) ? 97 : 0) );
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), gi );
   for( int gi = localRange.getEnd(); gi < localRange.getEnd() + overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), gi - ((rank == nproc-1) ? 97 : 0) );
}

TYPED_TEST( DistributedNDArrayOverlaps_1D_test, synchronize )
{
   test_helper_synchronize( this->distributedNDArray, this->rank, this->nproc );
}

#endif  // HAVE_GTEST


#if (defined(HAVE_GTEST) && defined(HAVE_MPI))
using CommunicatorType = Communicators::MpiCommunicator;

#include <sstream>

class MinimalistBufferedPrinter
: public ::testing::EmptyTestEventListener
{
private:
   std::stringstream sout;

public:
   // Called before a test starts.
   virtual void OnTestStart(const ::testing::TestInfo& test_info)
   {
      sout << test_info.test_case_name() << "." << test_info.name() << " Start." << std::endl;
   }

   // Called after a failed assertion or a SUCCEED() invocation.
   virtual void OnTestPartResult(const ::testing::TestPartResult& test_part_result)
   {
      sout << (test_part_result.failed() ? "====Failure=== " : "===Success=== ")
           << test_part_result.file_name() << " "
           << test_part_result.line_number() <<std::endl
           << test_part_result.summary() <<std::endl;
   }

   // Called after a test ends.
   virtual void OnTestEnd(const ::testing::TestInfo& test_info)
   {
      const int rank = CommunicatorType::GetRank(CommunicatorType::AllGroup);
      sout << test_info.test_case_name() << "." << test_info.name() << " End." <<std::endl;
      std::cout << rank << ":" << std::endl << sout.str()<< std::endl;
      sout.str( std::string() );
      sout.clear();
   }
};
#endif

#include "../../GtestMissingError.h"
int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );

   #ifdef HAVE_MPI
      ::testing::TestEventListeners& listeners =
         ::testing::UnitTest::GetInstance()->listeners();

      delete listeners.Release(listeners.default_result_printer());
      listeners.Append(new MinimalistBufferedPrinter);

      Communicators::ScopedInitializer< CommunicatorType > mpi(argc, argv);
   #endif
   return RUN_ALL_TESTS();
#else
   throw GtestMissingError();
#endif
}
