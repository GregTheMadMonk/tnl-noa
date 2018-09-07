/***************************************************************************
                          DistributedVectorTest.h  -  description
                             -------------------
    begin                : Sep 6, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

template< typename Vector >
void setLinearSequence( Vector& deviceVector )
{
   typename Vector::HostType a;
   a.setLike( deviceVector );
   for( int i = 0; i < a.getLocalVectorView().getSize(); i++ ) {
      const auto gi = a.getIndexMap().getGlobalIndex( i );
      a[ gi ] = gi;
   }
   deviceVector = a;
}

template< typename Vector >
void setConstantSequence( Vector& deviceVector,
                          typename Vector::RealType v )
{
   deviceVector.setValue( v );
}

template< typename Vector >
void setNegativeLinearSequence( Vector& deviceVector )
{
   typename Vector::HostType a;
   a.setLike( deviceVector );
   for( int i = 0; i < a.getLocalVectorView().getSize(); i++ ) {
      const auto gi = a.getIndexMap().getGlobalIndex( i );
      a[ gi ] = -gi;
   }
   deviceVector = a;
}

#ifdef HAVE_GTEST
#include <limits>

#include <gtest/gtest.h>

#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Communicators/NoDistrCommunicator.h>
#include <TNL/Communicators/ScopedInitializer.h>
#include <TNL/DistributedContainers/DistributedVector.h>
#include <TNL/DistributedContainers/Partitioner.h>

using namespace TNL;
using namespace TNL::DistributedContainers;

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
   using IndexMap = typename DistributedVector::IndexMapType;
   using DistributedVectorType = DistributedContainers::DistributedVector< RealType, DeviceType, CommunicatorType, IndexType, IndexMap >;
   using VectorViewType = typename DistributedVectorType::LocalVectorViewType;

   const int globalSize = 97;  // prime number to force non-uniform distribution

   const typename CommunicatorType::CommunicationGroup group = CommunicatorType::AllGroup;

   DistributedVectorType x, y, z;

   const int rank = CommunicatorType::GetRank(group);
   const int nproc = CommunicatorType::GetSize(group);

   DistributedVectorTest()
   {
      const IndexMap map = DistributedContainers::Partitioner< IndexMap, CommunicatorType >::splitRange( globalSize, group );
      x.setDistribution( map, group );
      y.setDistribution( map, group );
      z.setDistribution( map, group );

      setConstantSequence( x, 1 );
      setLinearSequence( y );
      setNegativeLinearSequence( z );
   }
};

// types for which DistributedVectorTest is instantiated
using DistributedVectorTypes = ::testing::Types<
   DistributedVector< double, Devices::Host, Communicators::MpiCommunicator, int, Subrange< int > >,
   DistributedVector< double, Devices::Host, Communicators::NoDistrCommunicator, int, Subrange< int > >
#ifdef HAVE_CUDA
   ,
   DistributedVector< double, Devices::Cuda, Communicators::MpiCommunicator, int, Subrange< int > >,
   DistributedVector< double, Devices::Cuda, Communicators::NoDistrCommunicator, int, Subrange< int > >
#endif
>;

TYPED_TEST_CASE( DistributedVectorTest, DistributedVectorTypes );

TYPED_TEST( DistributedVectorTest, max )
{
   EXPECT_EQ( this->x.max(), 1 );
   EXPECT_EQ( this->y.max(), this->globalSize - 1 );
   EXPECT_EQ( this->z.max(), 0 );
}

TYPED_TEST( DistributedVectorTest, min )
{
   EXPECT_EQ( this->x.min(), 1 );
   EXPECT_EQ( this->y.min(), 0 );
   EXPECT_EQ( this->z.min(), 1 - this->globalSize );
}

TYPED_TEST( DistributedVectorTest, absMax )
{
   EXPECT_EQ( this->x.absMax(), 1 );
   EXPECT_EQ( this->y.absMax(), this->globalSize - 1 );
   EXPECT_EQ( this->z.absMax(), this->globalSize - 1 );
}

TYPED_TEST( DistributedVectorTest, absMin )
{
   EXPECT_EQ( this->x.absMin(), 1 );
   EXPECT_EQ( this->y.absMin(), 0 );
   EXPECT_EQ( this->z.absMin(), 0 );
}

TYPED_TEST( DistributedVectorTest, lpNorm )
{
   using RealType = typename TestFixture::RealType;

   const RealType epsilon = 64 * std::numeric_limits< RealType >::epsilon();
   const RealType expectedL1norm = this->globalSize;
   const RealType expectedL2norm = std::sqrt( this->globalSize );
   const RealType expectedL3norm = std::cbrt( this->globalSize );
   EXPECT_EQ( this->x.lpNorm( 1.0 ), expectedL1norm );
   EXPECT_EQ( this->x.lpNorm( 2.0 ), expectedL2norm );
   EXPECT_NEAR( this->x.lpNorm( 3.0 ), expectedL3norm, epsilon );
}

TYPED_TEST( DistributedVectorTest, sum )
{
   EXPECT_EQ( this->x.sum(), this->globalSize );
   EXPECT_EQ( this->y.sum(), 0.5 * this->globalSize * ( this->globalSize - 1 ) );
   EXPECT_EQ( this->z.sum(), - 0.5 * this->globalSize * ( this->globalSize - 1 ) );
}

TYPED_TEST( DistributedVectorTest, differenceMax )
{
   EXPECT_EQ( this->x.differenceMax( this->y ), 1 );
   EXPECT_EQ( this->y.differenceMax( this->x ), this->globalSize - 2 );
}

TYPED_TEST( DistributedVectorTest, differenceMin )
{
   EXPECT_EQ( this->x.differenceMin( this->y ), 2 - this->globalSize );
   EXPECT_EQ( this->y.differenceMin( this->x ), -1 );
}

TYPED_TEST( DistributedVectorTest, differenceAbsMax )
{
   EXPECT_EQ( this->x.differenceAbsMax( this->y ), this->globalSize - 2 );
   EXPECT_EQ( this->y.differenceAbsMax( this->x ), this->globalSize - 2 );
}

TYPED_TEST( DistributedVectorTest, differenceAbsMin )
{
   EXPECT_EQ( this->x.differenceAbsMin( this->y ), 0 );
   EXPECT_EQ( this->y.differenceAbsMin( this->x ), 0 );
}

TYPED_TEST( DistributedVectorTest, differenceLpNorm )
{
   using RealType = typename TestFixture::RealType;

   this->x.setValue( 2 );
   this->y.setValue( 1 );
   const RealType epsilon = 64 * std::numeric_limits< RealType >::epsilon();
   const RealType expectedL1norm = this->globalSize;
   const RealType expectedL2norm = std::sqrt( this->globalSize );
   const RealType expectedL3norm = std::cbrt( this->globalSize );
   EXPECT_EQ( this->x.differenceLpNorm( this->y, 1.0 ), expectedL1norm );
   EXPECT_EQ( this->x.differenceLpNorm( this->y, 2.0 ), expectedL2norm );
   EXPECT_NEAR( this->x.differenceLpNorm( this->y, 3.0 ), expectedL3norm, epsilon );
}

TYPED_TEST( DistributedVectorTest, differenceSum )
{
   EXPECT_EQ( this->x.differenceSum( this->x ), 0 );
   EXPECT_EQ( this->y.differenceSum( this->x ), 0.5 * this->globalSize * ( this->globalSize - 1 ) - this->globalSize );
   EXPECT_EQ( this->y.differenceSum( this->y ), 0 );
}

TYPED_TEST( DistributedVectorTest, scalarMultiplication )
{
   this->y *= 2;
   for( int i = 0; i < this->y.getLocalVectorView().getSize(); i++ ) {
      const auto gi = this->y.getIndexMap().getGlobalIndex( i );
      EXPECT_EQ( this->y.getElement( gi ), 2 * gi );
   }

   this->y.scalarMultiplication( 2 );
   for( int i = 0; i < this->y.getLocalVectorView().getSize(); i++ ) {
      const auto gi = this->y.getIndexMap().getGlobalIndex( i );
      EXPECT_EQ( this->y.getElement( gi ), 4 * gi );
   }
}

TYPED_TEST( DistributedVectorTest, scalarProduct )
{
   EXPECT_EQ( this->x.scalarProduct( this->x ), this->globalSize );
   EXPECT_EQ( this->x.scalarProduct( this->y ), 0.5 * this->globalSize * ( this->globalSize - 1 ) );
   EXPECT_EQ( this->y.scalarProduct( this->x ), 0.5 * this->globalSize * ( this->globalSize - 1 ) );
   EXPECT_EQ( this->x.scalarProduct( this->z ), - 0.5 * this->globalSize * ( this->globalSize - 1 ) );
   EXPECT_EQ( this->z.scalarProduct( this->x ), - 0.5 * this->globalSize * ( this->globalSize - 1 ) );
}

TYPED_TEST( DistributedVectorTest, addVector )
{
   this->x.addVector( this->y, 3.0, 1.0 );
   for( int i = 0; i < this->y.getLocalVectorView().getSize(); i++ ) {
      const auto gi = this->y.getIndexMap().getGlobalIndex( i );
      EXPECT_EQ( this->x.getElement( gi ), 1 + 3 * gi );
   }

   this->y += this->z;
   for( int i = 0; i < this->y.getLocalVectorView().getSize(); i++ ) {
      const auto gi = this->y.getIndexMap().getGlobalIndex( i );
      EXPECT_EQ( this->y.getElement( gi ), 0 );
   }
}

TYPED_TEST( DistributedVectorTest, addVectors )
{
   this->x.addVectors( this->y, 3.0, this->z, 1.0, 2.0 );
   for( int i = 0; i < this->y.getLocalVectorView().getSize(); i++ ) {
      const auto gi = this->y.getIndexMap().getGlobalIndex( i );
      EXPECT_EQ( this->x.getElement( gi ), 2 + 3 * gi - gi );
   }
}

// TODO: distributed prefix sum

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
      int rank=CommunicatorType::GetRank(CommunicatorType::AllGroup);
      sout << test_info.test_case_name() << "." << test_info.name() << " End." <<std::endl;
      std::cout << rank << ":" << std::endl << sout.str()<< std::endl;
      sout.str( std::string() );
      sout.clear();
   }
};
#endif

#include "../GtestMissingError.h"
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
