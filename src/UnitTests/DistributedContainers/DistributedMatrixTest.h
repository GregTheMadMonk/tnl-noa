/***************************************************************************
                          DistributedMatrixTest.h  -  description
                             -------------------
    begin                : Sep 10, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

template< typename Vector >
void setLinearSequence( Vector& deviceVector, typename Vector::RealType offset = 0 )
{
   typename Vector::HostType a;
   a.setLike( deviceVector );
   for( int i = 0; i < a.getLocalVectorView().getSize(); i++ ) {
      const auto gi = a.getLocalRange().getGlobalIndex( i );
      a[ gi ] = gi + offset;
   }
   deviceVector = a;
}

template< typename Matrix, typename RowLengths >
void setMatrix( Matrix& matrix, const RowLengths& rowLengths )
{
   typename Matrix::HostType hostMatrix;
   typename RowLengths::HostType hostRowLengths;
   hostMatrix.setLike( matrix );
   hostRowLengths = rowLengths;
   hostMatrix.setCompressedRowLengths( hostRowLengths );

   for( int i = 0; i < hostMatrix.getLocalMatrix().getRows(); i++ ) {
      const auto gi = hostMatrix.getLocalRowRange().getGlobalIndex( i );
      for( int j = 0; j < hostRowLengths[ gi ]; j++ )
         hostMatrix.setElement( gi, hostMatrix.getColumns() - j - 1, 1 );
   }

   matrix = hostMatrix;
}

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Communicators/NoDistrCommunicator.h>
#include <TNL/Communicators/ScopedInitializer.h>
#include <TNL/DistributedContainers/DistributedMatrix.h>
#include <TNL/DistributedContainers/Partitioner.h>
#include <TNL/Matrices/CSR.h>

using namespace TNL;
using namespace TNL::DistributedContainers;

/*
 * Light check of DistributedMatrix.
 *
 * - Number of processes is not limited.
 * - Global size is hardcoded as 97 to force non-uniform distribution.
 * - Communication group is hardcoded as AllGroup -- it may be changed as needed.
 * - Matrix format is hardcoded as CSR -- it should be possible to change it to
 *   any other format which does not include padding zeros in the getRowLength()
 *   result.
 */
template< typename DistributedMatrix >
class DistributedMatrixTest
: public ::testing::Test
{
protected:
   using RealType = typename DistributedMatrix::RealType;
   using DeviceType = typename DistributedMatrix::DeviceType;
   using CommunicatorType = typename DistributedMatrix::CommunicatorType;
   using IndexType = typename DistributedMatrix::IndexType;
   using DistributedMatrixType = DistributedMatrix;

   using RowLengthsVector = typename DistributedMatrixType::CompressedRowLengthsVector;
   using GlobalVector = Containers::Vector< RealType, DeviceType, IndexType >;
   using DistributedVector = DistributedContainers::DistributedVector< RealType, DeviceType, IndexType, CommunicatorType >;

   const int globalSize = 97;  // prime number to force non-uniform distribution

   const typename CommunicatorType::CommunicationGroup group = CommunicatorType::AllGroup;

   const int rank = CommunicatorType::GetRank(group);
   const int nproc = CommunicatorType::GetSize(group);

   DistributedMatrixType matrix;

   RowLengthsVector rowLengths;

   DistributedMatrixTest()
   {
      using LocalRangeType = typename DistributedMatrix::LocalRangeType;
      const LocalRangeType localRange = DistributedContainers::Partitioner< IndexType, CommunicatorType >::splitRange( globalSize, group );
      matrix.setDistribution( localRange, globalSize, globalSize, group );
      rowLengths.setDistribution( localRange, globalSize, group );

      EXPECT_EQ( matrix.getLocalRowRange(), localRange );
      EXPECT_EQ( matrix.getCommunicationGroup(), group );

      setLinearSequence( rowLengths, 1 );
   }
};

// types for which DistributedMatrixTest is instantiated
using DistributedMatrixTypes = ::testing::Types<
   DistributedMatrix< Matrices::CSR< double, Devices::Host, int >, Communicators::MpiCommunicator >,
   DistributedMatrix< Matrices::CSR< double, Devices::Host, int >, Communicators::NoDistrCommunicator >
#ifdef HAVE_CUDA
   ,
   DistributedMatrix< Matrices::CSR< double, Devices::Cuda, int >, Communicators::MpiCommunicator >,
   DistributedMatrix< Matrices::CSR< double, Devices::Cuda, int >, Communicators::NoDistrCommunicator >
#endif
>;

TYPED_TEST_CASE( DistributedMatrixTest, DistributedMatrixTypes );

TYPED_TEST( DistributedMatrixTest, checkSumOfLocalSizes )
{
   using CommunicatorType = typename TestFixture::CommunicatorType;

   const int localSize = this->matrix.getLocalMatrix().getRows();
   int sumOfLocalSizes = 0;
   CommunicatorType::Allreduce( &localSize, &sumOfLocalSizes, 1, MPI_SUM, this->group );
   EXPECT_EQ( sumOfLocalSizes, this->globalSize );
   EXPECT_EQ( this->matrix.getRows(), this->globalSize );
}

TYPED_TEST( DistributedMatrixTest, setLike )
{
   using DistributedMatrixType = typename TestFixture::DistributedMatrixType;

   EXPECT_EQ( this->matrix.getRows(), this->globalSize );
   EXPECT_EQ( this->matrix.getColumns(), this->globalSize );
   DistributedMatrixType copy;
   EXPECT_EQ( copy.getRows(), 0 );
   EXPECT_EQ( copy.getColumns(), 0 );
   copy.setLike( this->matrix );
   EXPECT_EQ( copy.getRows(), this->globalSize );
   EXPECT_EQ( copy.getColumns(), this->globalSize );
}

TYPED_TEST( DistributedMatrixTest, reset )
{
   EXPECT_EQ( this->matrix.getRows(), this->globalSize );
   EXPECT_EQ( this->matrix.getColumns(), this->globalSize );
   EXPECT_GT( this->matrix.getLocalMatrix().getRows(), 0 );
   this->matrix.reset();
   EXPECT_EQ( this->matrix.getRows(), 0 );
   EXPECT_EQ( this->matrix.getColumns(), 0 );
   EXPECT_EQ( this->matrix.getLocalMatrix().getRows(), 0 );
}

TYPED_TEST( DistributedMatrixTest, setCompressedRowLengths )
{
   for( int i = 0; i < this->matrix.getLocalMatrix().getRows(); i++ ) {
      const auto gi = this->matrix.getLocalRowRange().getGlobalIndex( i );
      EXPECT_EQ( this->matrix.getRowLength( gi ), 0 );
      EXPECT_EQ( this->matrix.getLocalMatrix().getRowLength( i ), 0 );
   }
   this->matrix.setCompressedRowLengths( this->rowLengths );
   for( int i = 0; i < this->matrix.getLocalMatrix().getRows(); i++ ) {
      const auto gi = this->matrix.getLocalRowRange().getGlobalIndex( i );
      EXPECT_EQ( this->matrix.getRowLength( gi ), gi + 1 );
      EXPECT_EQ( this->matrix.getLocalMatrix().getRowLength( i ), gi + 1 );
   }
}

TYPED_TEST( DistributedMatrixTest, getCompressedRowLengths )
{
   using RowLengthsVector = typename TestFixture::RowLengthsVector;

   this->matrix.setCompressedRowLengths( this->rowLengths );
   RowLengthsVector output;
   this->matrix.getCompressedRowLengths( output );
   EXPECT_EQ( output, this->rowLengths );
}

TYPED_TEST( DistributedMatrixTest, setGetElement )
{
   // NOTE: the test is very slow for CUDA, but there is no reason it should fail
   // while it works for Host, so we just skip it.
   if( std::is_same< typename TestFixture::DeviceType, Devices::Cuda >::value )
      return;

   this->matrix.setCompressedRowLengths( this->rowLengths );
   for( int i = 0; i < this->matrix.getLocalMatrix().getRows(); i++ ) {
      const auto gi = this->matrix.getLocalRowRange().getGlobalIndex( i );
      for( int j = 0; j < this->rowLengths.getElement( gi ); j++ )
         this->matrix.setElement( gi, j,  gi + j );
   }
   for( int i = 0; i < this->matrix.getLocalMatrix().getRows(); i++ ) {
      const auto gi = this->matrix.getLocalRowRange().getGlobalIndex( i );
      for( int j = 0; j < this->rowLengths.getElement( gi ); j++ ) {
         EXPECT_EQ( this->matrix.getElement( gi, j ), gi + j );
         EXPECT_EQ( this->matrix.getLocalMatrix().getElement( i, j ), gi + j );
      }
   }
}

// TODO: setElementFast, getElementFast

// TODO: setRowFast, getRowFast

// TODO: getRow (const and non-const)

TYPED_TEST( DistributedMatrixTest, vectorProduct_globalInput )
{
   using GlobalVector = typename TestFixture::GlobalVector;
   using DistributedVector = typename TestFixture::DistributedVector;

   this->matrix.setCompressedRowLengths( this->rowLengths );
   setMatrix( this->matrix, this->rowLengths );

   GlobalVector inVector( this->globalSize );
   inVector.setValue( 1 );
   DistributedVector outVector( this->matrix.getLocalRowRange(), this->globalSize, this->matrix.getCommunicationGroup() );
   this->matrix.vectorProduct( inVector, outVector );

   EXPECT_EQ( outVector, this->rowLengths )
      << "outVector.getLocalVectorView() = " << outVector.getLocalVectorView()
      << ",\nthis->rowLengths.getLocalVectorView() = " << this->rowLengths.getLocalVectorView();
}

TYPED_TEST( DistributedMatrixTest, vectorProduct_distributedInput )
{
   using DistributedVector = typename TestFixture::DistributedVector;

   this->matrix.setCompressedRowLengths( this->rowLengths );
   setMatrix( this->matrix, this->rowLengths );

   DistributedVector inVector( this->matrix.getLocalRowRange(), this->globalSize, this->matrix.getCommunicationGroup() );
   inVector.setValue( 1 );
   DistributedVector outVector( this->matrix.getLocalRowRange(), this->globalSize, this->matrix.getCommunicationGroup() );
   this->matrix.vectorProduct( inVector, outVector );

   EXPECT_EQ( outVector, this->rowLengths )
      << "outVector.getLocalVectorView() = " << outVector.getLocalVectorView()
      << ",\nthis->rowLengths.getLocalVectorView() = " << this->rowLengths.getLocalVectorView();
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
