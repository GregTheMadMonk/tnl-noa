/***************************************************************************
                          DistributedNDArray_1D_test.h  -  description
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
class DistributedNDArray_1D_test
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

   const typename CommunicatorType::CommunicationGroup group = CommunicatorType::AllGroup;

   DistributedNDArrayType distributedNDArray;

   const int rank = CommunicatorType::GetRank(group);
   const int nproc = CommunicatorType::GetSize(group);

   DistributedNDArray_1D_test()
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

// types for which DistributedNDArray_1D_test is instantiated
using DistributedNDArrayTypes = ::testing::Types<
   DistributedNDArray< NDArray< double,
                                SizesHolder< int, 0 >,
                                std::index_sequence< 0 >,
                                std::index_sequence< 0 >,
                                Devices::Host >,
                       Communicators::MpiCommunicator >,
   DistributedNDArray< NDArray< double,
                                SizesHolder< int, 0 >,
                                std::index_sequence< 0 >,
                                std::index_sequence< 0 >,
                                Devices::Host >,
                       Communicators::NoDistrCommunicator >
#ifdef HAVE_CUDA
   ,
   DistributedNDArray< NDArray< double,
                                SizesHolder< int, 0 >,
                                std::index_sequence< 0 >,
                                std::index_sequence< 0 >,
                                Devices::Cuda >,
                       Communicators::MpiCommunicator >,
   DistributedNDArray< NDArray< double,
                                SizesHolder< int, 0 >,
                                std::index_sequence< 0 >,
                                std::index_sequence< 0 >,
                                Devices::Cuda >,
                       Communicators::NoDistrCommunicator >
#endif
>;

TYPED_TEST_SUITE( DistributedNDArray_1D_test, DistributedNDArrayTypes );

TYPED_TEST( DistributedNDArray_1D_test, checkSumOfLocalSizes )
{
   using CommunicatorType = typename TestFixture::CommunicatorType;

   const auto localRange = this->distributedNDArray.template getLocalRange< 0 >();
   const int localSize = localRange.getEnd() - localRange.getBegin();
   int sumOfLocalSizes = 0;
   CommunicatorType::Allreduce( &localSize, &sumOfLocalSizes, 1, MPI_SUM, this->group );
   EXPECT_EQ( sumOfLocalSizes, this->globalSize );
   EXPECT_EQ( this->distributedNDArray.template getSize< 0 >(), this->globalSize );
}

TYPED_TEST( DistributedNDArray_1D_test, setLike )
{
   using DistributedNDArrayType = typename TestFixture::DistributedNDArrayType;

   const auto localRange = this->distributedNDArray.template getLocalRange< 0 >();
   EXPECT_EQ( this->distributedNDArray.getLocalStorageSize(), localRange.getEnd() - localRange.getBegin() );
   DistributedNDArrayType copy;
   EXPECT_EQ( copy.getLocalStorageSize(), 0 );
   copy.setLike( this->distributedNDArray );
   EXPECT_EQ( copy.getLocalStorageSize(), localRange.getEnd() - localRange.getBegin() );
}

TYPED_TEST( DistributedNDArray_1D_test, reset )
{
   const auto localRange = this->distributedNDArray.template getLocalRange< 0 >();
   EXPECT_EQ( this->distributedNDArray.getLocalStorageSize(), localRange.getEnd() - localRange.getBegin() );
   this->distributedNDArray.reset();
   EXPECT_EQ( this->distributedNDArray.getLocalStorageSize(), 0 );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray, typename BufferView >
void test_helper_setValue( DistributedArray& array, BufferView& buffer_view )
{
   using DeviceType = typename DistributedArray::DeviceType;
   using IndexType = typename DistributedArray::IndexType;

   const auto localRange = array.template getLocalRange< 0 >();
   auto array_view = array.getConstView();
   auto kernel = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      buffer_view[ i - localRange.getBegin() ] = array_view( i );
   };
   ParallelFor< DeviceType >::exec( localRange.getBegin(), localRange.getEnd(), kernel );
}

TYPED_TEST( DistributedNDArray_1D_test, setValue )
{
   using LocalArrayType = typename TestFixture::LocalArrayType;
   using LocalArrayViewType = typename TestFixture::LocalArrayViewType;

   this->distributedNDArray.setValue( 1.0 );

   const auto localRange = this->distributedNDArray.template getLocalRange< 0 >();
   LocalArrayType buffer( localRange.getEnd() - localRange.getBegin() );
   LocalArrayViewType buffer_view( buffer );
   test_helper_setValue( this->distributedNDArray, buffer_view );

   LocalArrayType expected( localRange.getEnd() - localRange.getBegin() );
   expected.setValue( 1.0 );
   EXPECT_EQ( buffer, expected );
}

TYPED_TEST( DistributedNDArray_1D_test, elementwiseAccess )
{
//   using ArrayViewType = typename TestFixture::ArrayViewType;
   using IndexType = typename TestFixture::IndexType;

   this->distributedNDArray.setValue( 0 );
//   ArrayViewType localArrayView = this->distributedNDArray.getLocalArrayView();
   const auto localRange = this->distributedNDArray.template getLocalRange< 0 >();

   // check initial value
   for( IndexType gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ ) {
//      EXPECT_EQ( localArrayView.getElement( i ), 0 );
      EXPECT_EQ( this->distributedNDArray.getElement( gi ), 0 );
      if( std::is_same< typename TestFixture::DeviceType, Devices::Host >::value )
         EXPECT_EQ( this->distributedNDArray[ gi ], 0 );
   }

   // use operator()
   if( std::is_same< typename TestFixture::DeviceType, Devices::Host >::value ) {
      for( IndexType gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ ) {
         this->distributedNDArray( gi ) = gi + 1;
      }

      // check set value
      for( IndexType gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ ) {
//         EXPECT_EQ( localArrayView.getElement( i ), gi + 1 );
         EXPECT_EQ( this->distributedNDArray.getElement( gi ), gi + 1 );
         EXPECT_EQ( this->distributedNDArray( gi ), gi + 1 );
         EXPECT_EQ( this->distributedNDArray[ gi ], gi + 1 );
      }
   }
}

TYPED_TEST( DistributedNDArray_1D_test, copyAssignment )
{
   using DistributedNDArrayType = typename TestFixture::DistributedNDArrayType;

   this->distributedNDArray.setValue( 1 );
   DistributedNDArrayType copy;
   copy = this->distributedNDArray;
   // no binding, but deep copy
//   EXPECT_NE( copy.getLocalArrayView().getData(), this->distributedNDArray.getLocalArrayView().getData() );
//   EXPECT_EQ( copy.getLocalArrayView(), this->distributedNDArray.getLocalArrayView() );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_comparisonOperators( DistributedArray& u, DistributedArray& v, DistributedArray& w )
{
   using DeviceType = typename DistributedArray::DeviceType;
   using IndexType = typename DistributedArray::IndexType;

   const auto localRange = u.template getLocalRange< 0 >();
   auto u_view = u.getView();
   auto v_view = v.getView();
   auto w_view = w.getView();

   auto kernel = [=] __cuda_callable__ ( IndexType gi ) mutable
   {
      u_view( gi ) = gi;
      v_view( gi ) = gi;
      w_view( gi ) = 2 * gi;
   };
   ParallelFor< DeviceType >::exec( localRange.getBegin(), localRange.getEnd(), kernel );
}

TYPED_TEST( DistributedNDArray_1D_test, comparisonOperators )
{
   using DistributedNDArrayType = typename TestFixture::DistributedNDArrayType;

   DistributedNDArrayType& u = this->distributedNDArray;
   DistributedNDArrayType v, w;
   v.setLike( u );
   w.setLike( u );

   test_helper_comparisonOperators( u, v, w );

   EXPECT_TRUE( u == u );
   EXPECT_TRUE( u == v );
   EXPECT_TRUE( v == u );
   EXPECT_FALSE( u != v );
   EXPECT_FALSE( v != u );
   EXPECT_TRUE( u != w );
   EXPECT_TRUE( w != u );
   EXPECT_FALSE( u == w );
   EXPECT_FALSE( w == u );

   v.reset();
   EXPECT_FALSE( u == v );
   u.reset();
   EXPECT_TRUE( u == v );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_forAll( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const auto localRange = a.template getLocalRange< 0 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      a_view( i ) += 1;
   };

   a.setValue( 0 );
   a.forAll( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 );

   a.setValue( 0 );
   a_view.forAll( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 );
}

TYPED_TEST( DistributedNDArray_1D_test, forAll )
{
   test_helper_forAll( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_forInternal( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const auto localRange = a.template getLocalRange< 0 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      a_view( i ) += 1;
   };

   a.setValue( 0 );
   a.forInternal( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
   {
      if( gi == 0 || gi == a.template getSize< 0 >() - 1 )
         EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
      else
         EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
   }

   a.setValue( 0 );
   a_view.forInternal( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
   {
      if( gi == 0 || gi == a.template getSize< 0 >() - 1 )
         EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
      else
         EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
   }
}

TYPED_TEST( DistributedNDArray_1D_test, forInternal )
{
   test_helper_forInternal( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_forLocalInternal( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const auto localRange = a.template getLocalRange< 0 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      a_view( i ) += 1;
   };

   a.setValue( 0 );
   // equivalent to forAll because all overlaps are 0
   a.forLocalInternal( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;

   a.setValue( 0 );
   // equivalent to forAll because all overlaps are 0
   a_view.forLocalInternal( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
}

TYPED_TEST( DistributedNDArray_1D_test, forLocalInternal )
{
   test_helper_forLocalInternal( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_forBoundary( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const auto localRange = a.template getLocalRange< 0 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      a_view( i ) += 1;
   };

   a.setValue( 0 );
   a.forBoundary( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
   {
      if( gi == 0 || gi == a.template getSize< 0 >() - 1 )
         EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
      else
         EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
   }

   a.setValue( 0 );
   a_view.forBoundary( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
   {
      if( gi == 0 || gi == a.template getSize< 0 >() - 1 )
         EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
      else
         EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
   }
}

TYPED_TEST( DistributedNDArray_1D_test, forBoundary )
{
   test_helper_forBoundary( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_forLocalBoundary( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const auto localRange = a.template getLocalRange< 0 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      a_view( i ) += 1;
   };

   a.setValue( 0 );
   // empty set because all overlaps are 0
   a.forLocalBoundary( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;

   a.setValue( 0 );
   // empty set because all overlaps are 0
   a_view.forLocalBoundary( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
}

TYPED_TEST( DistributedNDArray_1D_test, forLocalBoundary )
{
   test_helper_forLocalBoundary( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_forOverlaps( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const auto localRange = a.template getLocalRange< 0 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      a_view( i ) += 1;
   };

   a.setValue( 0 );
   // empty set because all overlaps are 0
   a.forOverlaps( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;

   a.setValue( 0 );
   // empty set because all overlaps are 0
   a_view.forOverlaps( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
}

TYPED_TEST( DistributedNDArray_1D_test, forOverlaps )
{
   test_helper_forOverlaps( this->distributedNDArray );
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
