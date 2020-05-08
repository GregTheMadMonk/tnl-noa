/***************************************************************************
                          DistributedMatrixTest.h  -  description
                             -------------------
    begin                : Sep 10, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Communicators/NoDistrCommunicator.h>
#include <TNL/Matrices/DistributedMatrix.h>
#include <TNL/Containers/Partitioner.h>
#include <TNL/Matrices/SparseMatrix.h>

using namespace TNL;

template< typename Vector >
void setLinearSequence( Vector& deviceVector, typename Vector::RealType offset = 0 )
{
   using HostVector = typename Vector::template Self< typename Vector::RealType, TNL::Devices::Sequential >;
   HostVector a;
   a.setLike( deviceVector );
   for( int i = 0; i < a.getLocalView().getSize(); i++ ) {
      const auto gi = a.getLocalRange().getGlobalIndex( i );
      a[ gi ] = gi + offset;
   }
   deviceVector = a;
}

template< typename Matrix, typename RowLengths >
void setMatrix( Matrix& matrix, const RowLengths& rowLengths )
{
   using HostMatrix = Matrices::DistributedMatrix< typename Matrix::MatrixType::template Self< typename Matrix::RealType, TNL::Devices::Sequential >, typename Matrix::CommunicatorType >;
   using HostRowLengths = typename RowLengths::template Self< typename RowLengths::RealType, TNL::Devices::Sequential >;

   HostMatrix hostMatrix;
   HostRowLengths hostRowLengths;
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

/*
 * Light check of DistributedMatrix.
 *
 * - Number of processes is not limited.
 * - Global size is hardcoded as 97 to force non-uniform distribution.
 * - Communication group is hardcoded as AllGroup -- it may be changed as needed.
 * - Matrix format is hardcoded as CSR.
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
   using DistributedVector = Containers::DistributedVector< RealType, DeviceType, IndexType, CommunicatorType >;

   const int globalSize = 97;  // prime number to force non-uniform distribution

   const typename CommunicatorType::CommunicationGroup group = CommunicatorType::AllGroup;

   const int rank = CommunicatorType::GetRank(group);
   const int nproc = CommunicatorType::GetSize(group);

   DistributedMatrixType matrix;

   RowLengthsVector rowLengths;

   DistributedMatrixTest()
   {
      using LocalRangeType = typename DistributedMatrix::LocalRangeType;
      const LocalRangeType localRange = Containers::Partitioner< IndexType, CommunicatorType >::splitRange( globalSize, group );
      matrix.setDistribution( localRange, globalSize, globalSize, group );
      rowLengths.setDistribution( localRange, globalSize, group );

      EXPECT_EQ( matrix.getLocalRowRange(), localRange );
      EXPECT_EQ( matrix.getCommunicationGroup(), group );

      setLinearSequence( rowLengths, 1 );
   }
};

// types for which DistributedMatrixTest is instantiated
using DistributedMatrixTypes = ::testing::Types<
   Matrices::DistributedMatrix< Matrices::SparseMatrix< double, Devices::Host, int >, Communicators::MpiCommunicator >,
   Matrices::DistributedMatrix< Matrices::SparseMatrix< double, Devices::Host, int >, Communicators::NoDistrCommunicator >
#ifdef HAVE_CUDA
   ,
   Matrices::DistributedMatrix< Matrices::SparseMatrix< double, Devices::Cuda, int >, Communicators::MpiCommunicator >,
   Matrices::DistributedMatrix< Matrices::SparseMatrix< double, Devices::Cuda, int >, Communicators::NoDistrCommunicator >
#endif
>;

TYPED_TEST_SUITE( DistributedMatrixTest, DistributedMatrixTypes );
/*
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
      EXPECT_EQ( this->matrix.getRowCapacity( gi ), 0 );
      EXPECT_EQ( this->matrix.getLocalMatrix().getRowCapacity( i ), 0 );
   }
   this->matrix.setCompressedRowLengths( this->rowLengths );
   for( int i = 0; i < this->matrix.getLocalMatrix().getRows(); i++ ) {
      const auto gi = this->matrix.getLocalRowRange().getGlobalIndex( i );
      EXPECT_EQ( this->matrix.getRowCapacity( gi ), gi + 1 );
      EXPECT_EQ( this->matrix.getLocalMatrix().getRowCapacity( i ), gi + 1 );
   }
}

TYPED_TEST( DistributedMatrixTest, getCompressedRowLengths )
{
   using RowLengthsVector = typename TestFixture::RowLengthsVector;

   this->matrix.setCompressedRowLengths( this->rowLengths );
   RowLengthsVector output;
   this->matrix.getCompressedRowLengths( output );
   // zero row lengths because the matrix is empty
   EXPECT_EQ( output, 0 );
   for( int i = 0; i < this->matrix.getLocalMatrix().getRows(); i++ ) {
      const auto gi = this->matrix.getLocalRowRange().getGlobalIndex( i );
      output[ gi ] = this->matrix.getRowCapacity( gi );
   }
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
      << "outVector.getLocalView() = " << outVector.getLocalView()
      << ",\nthis->rowLengths.getLocalView() = " << this->rowLengths.getLocalView();
}
*/
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
      << "outVector.getLocalView() = " << outVector.getLocalView()
      << ",\nthis->rowLengths.getLocalView() = " << this->rowLengths.getLocalView();
}


#endif  // HAVE_GTEST

#include "../main_mpi.h"
