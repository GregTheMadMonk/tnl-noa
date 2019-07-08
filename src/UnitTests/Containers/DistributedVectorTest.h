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
      const auto gi = a.getLocalRange().getGlobalIndex( i );
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
      const auto gi = a.getLocalRange().getGlobalIndex( i );
      a[ gi ] = -gi;
   }
   deviceVector = a;
}

#ifdef HAVE_GTEST
#include <limits>

#include <gtest/gtest.h>

#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Communicators/NoDistrCommunicator.h>
#include <TNL/Containers/DistributedVector.h>
#include <TNL/Containers/DistributedVectorView.h>
#include <TNL/Containers/Partitioner.h>

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
   using VectorViewType = typename DistributedVectorType::LocalVectorViewType;
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

TYPED_TEST( DistributedVectorTest, max )
{
   EXPECT_EQ( max( this->x ), 1 );
   EXPECT_EQ( max( this->y ), this->globalSize - 1 );
   EXPECT_EQ( max( this->z ), 0 );

   EXPECT_EQ( max( this->x_view ), 1 );
   EXPECT_EQ( max( this->y_view ), this->globalSize - 1 );
   EXPECT_EQ( max( this->z_view ), 0 );
}

TYPED_TEST( DistributedVectorTest, min )
{
   EXPECT_EQ( min( this->x ), 1 );
   EXPECT_EQ( min( this->y ), 0 );
   EXPECT_EQ( min( this->z ), 1 - this->globalSize );

   EXPECT_EQ( min( this->x_view ), 1 );
   EXPECT_EQ( min( this->y_view ), 0 );
   EXPECT_EQ( min( this->z_view ), 1 - this->globalSize );
}

TYPED_TEST( DistributedVectorTest, absMax )
{
   TNL::min( abs( this->x ) );
   /*EXPECT_EQ( TNL::max( abs( this->x ) ), 1 );
   EXPECT_EQ( max( abs( this->y ) ), this->globalSize - 1 );
   EXPECT_EQ( max( abs( this->z ) ), this->globalSize - 1 );

   EXPECT_EQ( max( abs( this->x_view ) ), 1 );
   EXPECT_EQ( max( abs( this->y_view ) ), this->globalSize - 1 );
   EXPECT_EQ( max( abs( this->z_view ) ), this->globalSize - 1 );*/
}

TYPED_TEST( DistributedVectorTest, absMin )
{
   EXPECT_EQ( min( abs( this->x ) ), 1 );
   EXPECT_EQ( min( abs( this->y ) ), 0 );
   EXPECT_EQ( min( abs( this->z ) ), 0 );

   EXPECT_EQ( min( abs( this->x_view ) ), 1 );
   EXPECT_EQ( min( abs( this->y_view ) ), 0 );
   EXPECT_EQ( min( abs( this->z_view ) ), 0 );
}

TYPED_TEST( DistributedVectorTest, lpNorm )
{
   using RealType = typename TestFixture::RealType;

   const RealType epsilon = 64 * std::numeric_limits< RealType >::epsilon();
   const RealType expectedL1norm = this->globalSize;
   const RealType expectedL2norm = std::sqrt( this->globalSize );
   const RealType expectedL3norm = std::cbrt( this->globalSize );

   EXPECT_EQ( lpNorm( this->x , 1.0 ), expectedL1norm );
   EXPECT_EQ( lpNorm( this->x,  2.0 ), expectedL2norm );
   EXPECT_NEAR( lpNorm( this->x,  3.0 ), expectedL3norm, epsilon );

   EXPECT_EQ( lpNorm( this->x_view, 1.0 ), expectedL1norm );
   EXPECT_EQ( lpNorm( this->x_view, 2.0 ), expectedL2norm );
   EXPECT_NEAR( lpNorm( this->x_view, 3.0 ), expectedL3norm, epsilon );
}

TYPED_TEST( DistributedVectorTest, sum )
{
   EXPECT_EQ( sum( this->x ), this->globalSize );
   EXPECT_EQ( sum( this->y ), 0.5 * this->globalSize * ( this->globalSize - 1 ) );
   EXPECT_EQ( sum( this->z ), - 0.5 * this->globalSize * ( this->globalSize - 1 ) );

   EXPECT_EQ( sum( this->x_view ), this->globalSize );
   EXPECT_EQ( sum( this->y_view ), 0.5 * this->globalSize * ( this->globalSize - 1 ) );
   EXPECT_EQ( sum( this->z_view ), - 0.5 * this->globalSize * ( this->globalSize - 1 ) );
}

TYPED_TEST( DistributedVectorTest, differenceMax )
{
   EXPECT_TRUE( max( this->x, this->y ) == 1 );
   EXPECT_TRUE( max( this->y - this->x ) == this->globalSize - 2 );

   EXPECT_EQ( max( this->x_view - this->y_view ), 1 );
   EXPECT_EQ( max( this->y_view - this->x_view ), this->globalSize - 2 );
}

TYPED_TEST( DistributedVectorTest, differenceMin )
{
   EXPECT_EQ( min( this->x - this->y ), 2 - this->globalSize );
   EXPECT_EQ( min( this->y - this->x ), -1 );

   EXPECT_EQ( min( this->x_view - this->y_view ), 2 - this->globalSize );
   EXPECT_EQ( min( this->y_view - this->x_view ), -1 );
}

TYPED_TEST( DistributedVectorTest, differenceAbsMax )
{
   EXPECT_EQ( max( abs( this->x - this->y ) ), this->globalSize - 2 );
   EXPECT_EQ( max( abs( this->y - this->x ) ), this->globalSize - 2 );

   EXPECT_EQ( max( abs( this->x_view - this->y_view ) ), this->globalSize - 2 );
   EXPECT_EQ( max( abs( this->y_view - this->x_view ) ), this->globalSize - 2 );
}

TYPED_TEST( DistributedVectorTest, differenceAbsMin )
{
   EXPECT_EQ( min( abs( this->x - this->y ) ), 0 );
   EXPECT_EQ( min( abs( this->y - this->x ) ), 0 );

   EXPECT_EQ( min( abs( this->x_view - this->y_view ) ), 0 );
   EXPECT_EQ( min( abs( this->y_view - this->x_view ) ), 0 );
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

   EXPECT_EQ( lpNorm( this->x - this->y, 1.0 ), expectedL1norm );
   EXPECT_EQ( lpNorm( this->x -  this->y, 2.0 ), expectedL2norm );
   EXPECT_NEAR( lpNorm( this->x - this->y, 3.0 ), expectedL3norm, epsilon );

   EXPECT_EQ( lpNorm( this->x_view - this->y_view, 1.0 ), expectedL1norm );
   EXPECT_EQ( lpNorm( this->x_view - this->y_view, 2.0 ), expectedL2norm );
   EXPECT_NEAR( lpNorm( this->x_view - this->y_view, 3.0 ), expectedL3norm, epsilon );
}

TYPED_TEST( DistributedVectorTest, differenceSum )
{
   EXPECT_EQ( TNL::sum( this->x - this->x ), 0 );
   EXPECT_EQ( TNL::sum( this->y - this->x ), 0.5 * this->globalSize * ( this->globalSize - 1 ) - this->globalSize );
   EXPECT_EQ( TNL::sum( this->y - this->y ), 0 );

   EXPECT_EQ( TNL::sum( this->x_view - this->x_view ), 0 );
   EXPECT_EQ( TNL::sum( this->y_view - this->x_view ), 0.5 * this->globalSize * ( this->globalSize - 1 ) - this->globalSize );
   EXPECT_EQ( TNL::sum( this->y_view - this->y_view ), 0 );
}

TYPED_TEST( DistributedVectorTest, scalarMultiplication )
{
   this->y *= 2;
   for( int i = 0; i < this->y.getLocalVectorView().getSize(); i++ ) {
      const auto gi = this->y.getLocalRange().getGlobalIndex( i );
      EXPECT_EQ( this->y.getElement( gi ), 2 * gi );
   }

   this->y.scalarMultiplication( 2 );
   for( int i = 0; i < this->y.getLocalVectorView().getSize(); i++ ) {
      const auto gi = this->y.getLocalRange().getGlobalIndex( i );
      EXPECT_EQ( this->y.getElement( gi ), 4 * gi );
   }

   setLinearSequence( this->y );

   this->y_view *= 2;
   for( int i = 0; i < this->y_view.getLocalVectorView().getSize(); i++ ) {
      const auto gi = this->y_view.getLocalRange().getGlobalIndex( i );
      EXPECT_EQ( this->y_view.getElement( gi ), 2 * gi );
   }

   this->y_view.scalarMultiplication( 2 );
   for( int i = 0; i < this->y_view.getLocalVectorView().getSize(); i++ ) {
      const auto gi = this->y_view.getLocalRange().getGlobalIndex( i );
      EXPECT_EQ( this->y_view.getElement( gi ), 4 * gi );
   }
}

TYPED_TEST( DistributedVectorTest, scalarProduct )
{
   EXPECT_EQ( this->x.scalarProduct( this->x ), this->globalSize );
   EXPECT_EQ( this->x.scalarProduct( this->y ), 0.5 * this->globalSize * ( this->globalSize - 1 ) );
   EXPECT_EQ( this->y.scalarProduct( this->x ), 0.5 * this->globalSize * ( this->globalSize - 1 ) );
   EXPECT_EQ( this->x.scalarProduct( this->z ), - 0.5 * this->globalSize * ( this->globalSize - 1 ) );
   EXPECT_EQ( this->z.scalarProduct( this->x ), - 0.5 * this->globalSize * ( this->globalSize - 1 ) );

   EXPECT_EQ( this->x_view.scalarProduct( this->x_view ), this->globalSize );
   EXPECT_EQ( this->x_view.scalarProduct( this->y_view ), 0.5 * this->globalSize * ( this->globalSize - 1 ) );
   EXPECT_EQ( this->y_view.scalarProduct( this->x_view ), 0.5 * this->globalSize * ( this->globalSize - 1 ) );
   EXPECT_EQ( this->x_view.scalarProduct( this->z_view ), - 0.5 * this->globalSize * ( this->globalSize - 1 ) );
   EXPECT_EQ( this->z_view.scalarProduct( this->x_view ), - 0.5 * this->globalSize * ( this->globalSize - 1 ) );
}

TYPED_TEST( DistributedVectorTest, addVector )
{
   this->x.addVector( this->y, 3.0, 1.0 );
   for( int i = 0; i < this->y.getLocalVectorView().getSize(); i++ ) {
      const auto gi = this->y.getLocalRange().getGlobalIndex( i );
      EXPECT_EQ( this->x.getElement( gi ), 1 + 3 * gi );
   }

   this->y += this->z;
   for( int i = 0; i < this->y.getLocalVectorView().getSize(); i++ ) {
      const auto gi = this->y.getLocalRange().getGlobalIndex( i );
      EXPECT_EQ( this->y.getElement( gi ), 0 );
   }

   setConstantSequence( this->x, 1 );
   setLinearSequence( this->y );

   this->x_view.addVector( this->y_view, 3.0, 1.0 );
   for( int i = 0; i < this->y_view.getLocalVectorView().getSize(); i++ ) {
      const auto gi = this->y_view.getLocalRange().getGlobalIndex( i );
      EXPECT_EQ( this->x_view.getElement( gi ), 1 + 3 * gi );
   }

   this->y_view += this->z_view;
   for( int i = 0; i < this->y_view.getLocalVectorView().getSize(); i++ ) {
      const auto gi = this->y_view.getLocalRange().getGlobalIndex( i );
      EXPECT_EQ( this->y_view.getElement( gi ), 0 );
   }
}

TYPED_TEST( DistributedVectorTest, addVectors )
{
   this->x.addVectors( this->y, 3.0, this->z, 1.0, 2.0 );
   for( int i = 0; i < this->y.getLocalVectorView().getSize(); i++ ) {
      const auto gi = this->y.getLocalRange().getGlobalIndex( i );
      EXPECT_EQ( this->x.getElement( gi ), 2 + 3 * gi - gi );
   }

   setConstantSequence( this->x, 1 );

   this->x_view.addVectors( this->y_view, 3.0, this->z_view, 1.0, 2.0 );
   for( int i = 0; i < this->y_view.getLocalVectorView().getSize(); i++ ) {
      const auto gi = this->y_view.getLocalRange().getGlobalIndex( i );
      EXPECT_EQ( this->x_view.getElement( gi ), 2 + 3 * gi - gi );
   }
}

// TODO: distributed prefix sum

#endif  // HAVE_GTEST

#include "../main_mpi.h"
