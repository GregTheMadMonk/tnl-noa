/***************************************************************************
                          DistributedArrayTest.h  -  description
                             -------------------
    begin                : Sep 6, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Containers/DistributedArray.h>
#include <TNL/Containers/Partitioner.h>

using namespace TNL;
using namespace TNL::Containers;

/*
 * Light check of DistributedArray.
 *
 * - Number of processes is not limited.
 * - Global size is hardcoded as 97 to force non-uniform distribution.
 * - Communication group is hardcoded as AllGroup -- it may be changed as needed.
 */
template< typename DistributedArray >
class DistributedArrayTest
: public ::testing::Test
{
protected:
   using ValueType = typename DistributedArray::ValueType;
   using DeviceType = typename DistributedArray::DeviceType;
   using CommunicatorType = typename DistributedArray::CommunicatorType;
   using IndexType = typename DistributedArray::IndexType;
   using DistributedArrayType = DistributedArray;
   using ArrayViewType = typename DistributedArrayType::LocalViewType;
   using ArrayType = Array< typename ArrayViewType::ValueType, typename ArrayViewType::DeviceType, typename ArrayViewType::IndexType >;

   const int globalSize = 97;  // prime number to force non-uniform distribution

   const typename CommunicatorType::CommunicationGroup group = CommunicatorType::AllGroup;

   DistributedArrayType distributedArray;

   const int rank = CommunicatorType::GetRank(group);
   const int nproc = CommunicatorType::GetSize(group);

   DistributedArrayTest()
   {
      using LocalRangeType = typename DistributedArray::LocalRangeType;
      const LocalRangeType localRange = Partitioner< IndexType, CommunicatorType >::splitRange( globalSize, group );
      distributedArray.setDistribution( localRange, globalSize, group );

      EXPECT_EQ( distributedArray.getLocalRange(), localRange );
      EXPECT_EQ( distributedArray.getCommunicationGroup(), group );
   }
};

// types for which DistributedArrayTest is instantiated
using DistributedArrayTypes = ::testing::Types<
   DistributedArray< double, Devices::Host, int, Communicators::MpiCommunicator >
#ifdef HAVE_CUDA
   ,
   DistributedArray< double, Devices::Cuda, int, Communicators::MpiCommunicator >
#endif
>;

TYPED_TEST_SUITE( DistributedArrayTest, DistributedArrayTypes );

TYPED_TEST( DistributedArrayTest, checkSumOfLocalSizes )
{
   using CommunicatorType = typename TestFixture::CommunicatorType;

   const int localSize = this->distributedArray.getLocalView().getSize();
   int sumOfLocalSizes = 0;
   CommunicatorType::Allreduce( &localSize, &sumOfLocalSizes, 1, MPI_SUM, this->group );
   EXPECT_EQ( sumOfLocalSizes, this->globalSize );
   EXPECT_EQ( this->distributedArray.getSize(), this->globalSize );
}

TYPED_TEST( DistributedArrayTest, copyFromGlobal )
{
   using ArrayViewType = typename TestFixture::ArrayViewType;
   using ArrayType = typename TestFixture::ArrayType;

   this->distributedArray.setValue( 0.0 );
   ArrayType globalArray( this->globalSize );
   globalArray.setValue( 1.0 );
   this->distributedArray.copyFromGlobal( globalArray );

   ArrayViewType localArrayView = this->distributedArray.getLocalView();
   auto globalView = globalArray.getConstView();
   const auto localRange = this->distributedArray.getLocalRange();
   globalView.bind( &globalArray.getData()[ localRange.getBegin() ], localRange.getEnd() - localRange.getBegin() );
   EXPECT_EQ( localArrayView, globalView );
}

TYPED_TEST( DistributedArrayTest, setLike )
{
   using DistributedArrayType = typename TestFixture::DistributedArrayType;

   EXPECT_EQ( this->distributedArray.getSize(), this->globalSize );
   DistributedArrayType copy;
   EXPECT_EQ( copy.getSize(), 0 );
   copy.setLike( this->distributedArray );
   EXPECT_EQ( copy.getSize(), this->globalSize );
}

TYPED_TEST( DistributedArrayTest, reset )
{
   EXPECT_EQ( this->distributedArray.getSize(), this->globalSize );
   EXPECT_GT( this->distributedArray.getLocalView().getSize(), 0 );
   this->distributedArray.reset();
   EXPECT_EQ( this->distributedArray.getSize(), 0 );
   EXPECT_EQ( this->distributedArray.getLocalView().getSize(), 0 );
}

// TODO: swap

TYPED_TEST( DistributedArrayTest, setValue )
{
   using ArrayViewType = typename TestFixture::ArrayViewType;
   using ArrayType = typename TestFixture::ArrayType;

   this->distributedArray.setValue( 1.0 );
   ArrayViewType localArrayView = this->distributedArray.getLocalView();
   ArrayType expected( localArrayView.getSize() );
   expected.setValue( 1.0 );
   EXPECT_EQ( localArrayView, expected );
}

TYPED_TEST( DistributedArrayTest, elementwiseAccess )
{
   using ArrayViewType = typename TestFixture::ArrayViewType;
   using IndexType = typename TestFixture::IndexType;

   this->distributedArray.setValue( 0 );
   ArrayViewType localArrayView = this->distributedArray.getLocalView();
   const auto localRange = this->distributedArray.getLocalRange();

   // check initial value
   for( IndexType i = 0; i < localArrayView.getSize(); i++ ) {
      const IndexType gi = localRange.getGlobalIndex( i );
      EXPECT_EQ( localArrayView.getElement( i ), 0 );
      EXPECT_EQ( this->distributedArray.getElement( gi ), 0 );
      if( std::is_same< typename TestFixture::DeviceType, Devices::Host >::value ) {
         EXPECT_EQ( this->distributedArray[ gi ], 0 );
      }
   }

   // use setValue
   for( IndexType i = 0; i < localArrayView.getSize(); i++ ) {
      const IndexType gi = localRange.getGlobalIndex( i );
      this->distributedArray.setElement( gi, i + 1 );
   }

   // check set value
   for( IndexType i = 0; i < localArrayView.getSize(); i++ ) {
      const IndexType gi = localRange.getGlobalIndex( i );
      EXPECT_EQ( localArrayView.getElement( i ), i + 1 );
      EXPECT_EQ( this->distributedArray.getElement( gi ), i + 1 );
      if( std::is_same< typename TestFixture::DeviceType, Devices::Host >::value ) {
         EXPECT_EQ( this->distributedArray[ gi ], i + 1 );
      }
   }

   this->distributedArray.setValue( 0 );

   // use operator[]
   if( std::is_same< typename TestFixture::DeviceType, Devices::Host >::value ) {
      for( IndexType i = 0; i < localArrayView.getSize(); i++ ) {
         const IndexType gi = localRange.getGlobalIndex( i );
         this->distributedArray[ gi ] = i + 1;
      }

      // check set value
      for( IndexType i = 0; i < localArrayView.getSize(); i++ ) {
         const IndexType gi = localRange.getGlobalIndex( i );
         EXPECT_EQ( localArrayView.getElement( i ), i + 1 );
         EXPECT_EQ( this->distributedArray.getElement( gi ), i + 1 );
         EXPECT_EQ( this->distributedArray[ gi ], i + 1 );
      }
   }
}

TYPED_TEST( DistributedArrayTest, copyConstructor )
{
   using DistributedArrayType = typename TestFixture::DistributedArrayType;

   this->distributedArray.setValue( 1 );
   DistributedArrayType copy( this->distributedArray );
   // Array has "binding" copy-constructor
   //EXPECT_EQ( copy.getLocalView().getData(), this->distributedArray.getLocalView().getData() );
}

TYPED_TEST( DistributedArrayTest, copyAssignment )
{
   using DistributedArrayType = typename TestFixture::DistributedArrayType;

   this->distributedArray.setValue( 1 );
   DistributedArrayType copy;
   copy = this->distributedArray;
   // no binding, but deep copy
   EXPECT_NE( copy.getLocalView().getData(), this->distributedArray.getLocalView().getData() );
   EXPECT_EQ( copy.getLocalView(), this->distributedArray.getLocalView() );
}

TYPED_TEST( DistributedArrayTest, comparisonOperators )
{
   using DistributedArrayType = typename TestFixture::DistributedArrayType;
   using IndexType = typename TestFixture::IndexType;

   const auto localRange = this->distributedArray.getLocalRange();
   DistributedArrayType& u = this->distributedArray;
   DistributedArrayType v, w;
   v.setLike( u );
   w.setLike( u );

   for( int i = 0; i < u.getLocalView().getSize(); i ++ ) {
      const IndexType gi = localRange.getGlobalIndex( i );
      u.setElement( gi, i );
      v.setElement( gi, i );
      w.setElement( gi, 2 * i );
   }

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

TYPED_TEST( DistributedArrayTest, containsValue )
{
   using IndexType = typename TestFixture::IndexType;

   const auto localRange = this->distributedArray.getLocalRange();

   for( int i = 0; i < this->distributedArray.getLocalView().getSize(); i++ ) {
      const IndexType gi = localRange.getGlobalIndex( i );
      this->distributedArray.setElement( gi, i % 10 );
   }

   for( int i = 0; i < 10; i++ )
      EXPECT_TRUE( this->distributedArray.containsValue( i ) );

   for( int i = 10; i < 20; i++ )
      EXPECT_FALSE( this->distributedArray.containsValue( i ) );
}

TYPED_TEST( DistributedArrayTest, containsOnlyValue )
{
   using IndexType = typename TestFixture::IndexType;

   const auto localRange = this->distributedArray.getLocalRange();

   for( int i = 0; i < this->distributedArray.getLocalView().getSize(); i++ ) {
      const IndexType gi = localRange.getGlobalIndex( i );
      this->distributedArray.setElement( gi, i % 10 );
   }

   for( int i = 0; i < 20; i++ )
      EXPECT_FALSE( this->distributedArray.containsOnlyValue( i ) );

   this->distributedArray.setValue( 100 );
   EXPECT_TRUE( this->distributedArray.containsOnlyValue( 100 ) );
}

TYPED_TEST( DistributedArrayTest, empty )
{
   EXPECT_GT( this->distributedArray.getSize(), 0 );
   EXPECT_FALSE( this->distributedArray.empty() );
   this->distributedArray.reset();
   EXPECT_EQ( this->distributedArray.getSize(), 0 );
   EXPECT_TRUE( this->distributedArray.empty() );
}

#endif  // HAVE_GTEST

#include "../main_mpi.h"
