/***************************************************************************
                          VectorOperationsTester.h  -  description
                             -------------------
    begin                : Mar 30, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#ifdef HAVE_GTEST 
#include "gtest/gtest.h"
#endif

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorOperations.h>

using namespace TNL;
using namespace TNL::Containers;

#ifdef HAVE_GTEST

typedef double Real;
typedef Devices::Host Device;
typedef int Index;

template< typename Vector >
void setLinearSequence( Vector& deviceVector )
{
   Containers::Vector< typename Vector :: RealType, Devices::Host > a;
   a. setSize( deviceVector. getSize() );
   for( int i = 0; i < a. getSize(); i ++ )
      a. getData()[ i ] = i;

   ArrayOperations< typename Vector::DeviceType,
                       Devices::Host >::
   template copyMemory< typename Vector::RealType,
                        typename Vector::RealType,
                        typename Vector::IndexType >
                      ( deviceVector.getData(),
                        a.getData(),
                        a.getSize() );
}


template< typename Vector >
void setOnesSequence( Vector& deviceVector )
{
   Containers::Vector< typename Vector :: RealType, Devices::Host > a;
   a. setSize( deviceVector. getSize() );
   for( int i = 0; i < a. getSize(); i ++ )
      a. getData()[ i ] = 1;

   ArrayOperations< typename Vector::DeviceType,
                       Devices::Host >::
   template copyMemory< typename Vector::RealType,
                        typename Vector::RealType,
                        typename Vector::IndexType >
                      ( deviceVector.getData(),
                        a.getData(),
                        a.getSize() );
}


template< typename Vector >
void setNegativeLinearSequence( Vector& deviceVector )
{
   Containers::Vector< typename Vector :: RealType, Devices::Host > a;
   a. setSize( deviceVector. getSize() );
   for( int i = 0; i < a. getSize(); i ++ )
      a. getData()[ i ] = -i;

   ArrayOperations< typename Vector::DeviceType,
                       Devices::Host >::
   template copyMemory< typename Vector::RealType,
                        typename Vector::RealType,
                        typename Vector::IndexType >
                      ( deviceVector.getData(),
                        a.getData(),
                        a.getSize() );
}

template< typename Vector >
void setOscilatingSequence( Vector& deviceVector,
                            typename Vector::RealType v )
{
   Containers::Vector< typename Vector::RealType, Devices::Host > a;
   a.setSize( deviceVector. getSize() );
   a[ 0 ] = v;
   for( int i = 1; i < a. getSize(); i ++ )
      a.getData()[ i ] = a.getData()[ i-1 ] * -1;

   ArrayOperations< typename Vector::DeviceType,
                       Devices::Host >::
   template copyMemory< typename Vector::RealType,
                        typename Vector::RealType,
                        typename Vector::IndexType >
                      ( deviceVector.getData(),
                        a.getData(),
                        a.getSize() );
}



TEST( VectorOperationsTest, getVectorMaxTest )
{
   const int size( 123456 );
   Containers::Vector< Real, Device > v;
   v. setSize( size );
   setLinearSequence( v );

   ASSERT_TRUE( Containers::VectorOperations< Device > :: getVectorMax( v ) == size - 1 );
}

TEST( VectorOperationsTest, getVectorMinTest )
{
   const int size( 123456 );
   Containers::Vector< Real, Device > v;
   v. setSize( size );
   setLinearSequence( v );

   ASSERT_TRUE( Containers::VectorOperations< Device > :: getVectorMin( v ) == 0 );
}

TEST( VectorOperationsTest, getVectorAbsMaxTest )
{
   const int size( 123456 );
   Containers::Vector< Real, Device > v;
   v. setSize( size );
   setNegativeLinearSequence( v );

   ASSERT_TRUE( Containers::VectorOperations< Device > :: getVectorAbsMax( v ) == size - 1 );
}

TEST( VectorOperationsTest, getVectorAbsMinTest )
{
   const int size( 123456 );
   Containers::Vector< Real, Device > v;
   v. setSize( size );
   setNegativeLinearSequence( v );

   ASSERT_TRUE( Containers::VectorOperations< Device > :: getVectorAbsMin( v ) == 0 );
}

TEST( VectorOperationsTest, getVectorLpNormTest )
{
   const int size( 123456 );
   Containers::Vector< Real, Device > v;
   v. setSize( size );
   setOnesSequence( v );

   ASSERT_TRUE( Containers::VectorOperations< Device > :: getVectorLpNorm( v, 2.0 ) == ::sqrt( size ) );
}

TEST( VectorOperationsTest, getVectorSumTest )
{
   const int size( 123456 );
   Containers::Vector< Real, Device > v;
   v. setSize( size );
   setOnesSequence( v );

   ASSERT_TRUE( Containers::VectorOperations< Device > :: getVectorSum( v ) == size );

   setLinearSequence( v );

   ASSERT_TRUE( Containers::VectorOperations< Device > :: getVectorSum( v ) == ( ( Real ) size ) * ( ( Real ) size - 1 ) / 2 );
}

TEST( VectorOperationsTest, getVectorDifferenceMaxTest )
{
   const int size( 123456 );
   Containers::Vector< Real, Device > u, v;
   u. setSize( size );
   v. setSize( size );
   setLinearSequence( u );
   setOnesSequence( v );

   ASSERT_TRUE( Containers::VectorOperations< Device > :: getVectorDifferenceMax( u, v ) == size - 2 );
}

TEST( VectorOperationsTest, getVectorDifferenceMinTest )
{
   const int size( 123456 );
   Containers::Vector< Real, Device > u, v;
   u. setSize( size );
   v. setSize( size );
   setLinearSequence( u );
   setOnesSequence( v );

   ASSERT_TRUE( Containers::VectorOperations< Device > :: getVectorDifferenceMin( u, v ) == -1 );
   ASSERT_TRUE( Containers::VectorOperations< Device > :: getVectorDifferenceMin( v, u ) == -123454 );
}

TEST( VectorOperationsTest, getVectorDifferenceAbsMaxTest )
{
   const int size( 123456 );
   Containers::Vector< Real, Device > u, v;
   u. setSize( size );
   v. setSize( size );
   setNegativeLinearSequence( u );
   setOnesSequence( v );

   ASSERT_TRUE( Containers::VectorOperations< Device > :: getVectorDifferenceAbsMax( u, v ) == size );
}

TEST( VectorOperationsTest, getVectorDifferenceAbsMinTest )
{
   const int size( 123456 );
   Containers::Vector< Real, Device > u, v;
   u. setSize( size );
   v. setSize( size );
   setLinearSequence( u );
   setOnesSequence( v );

   ASSERT_TRUE( Containers::VectorOperations< Device > :: getVectorDifferenceAbsMin( u, v ) == 0 );
   ASSERT_TRUE( Containers::VectorOperations< Device > :: getVectorDifferenceAbsMin( v, u ) == 0 );
}

TEST( VectorOperationsTest, getVectorDifferenceLpNormTest )
{
   const int size( 1024 );
   Containers::Vector< Real, Device > u, v;
   u. setSize( size );
   v. setSize( size );
   u. setValue( 3.0 );
   v. setValue( 1.0 );

   ASSERT_TRUE( Containers::VectorOperations< Device > :: getVectorDifferenceLpNorm( u, v, 1.0 ) == 2.0 * size );
   ASSERT_TRUE( Containers::VectorOperations< Device > :: getVectorDifferenceLpNorm( u, v, 2.0 ) == ::sqrt( 4.0 * size ) );
}

TEST( VectorOperationsTest, getVectorDifferenceSumTest )
{
   const int size( 1024 );
   Containers::Vector< Real, Device > u, v;
   u. setSize( size );
   v. setSize( size );
   u. setValue( 3.0 );
   v. setValue( 1.0 );

   ASSERT_TRUE( Containers::VectorOperations< Device > :: getVectorDifferenceSum( u, v ) == 2.0 * size );
}

TEST( VectorOperationsTest, vectorScalarMultiplicationTest )
{
   const int size( 1025 );
   Containers::Vector< Real, Device > u;
   u. setSize( size );
   setLinearSequence( u );

   Containers::VectorOperations< Device >::vectorScalarMultiplication( u, 3.0 );

   for( int i = 0; i < size; i++ )
      ASSERT_TRUE( u.getElement( i ) == 3.0 * i );
}

TEST( VectorOperationsTest, getVectorScalarProductTest )
{
   const int size( 1025 );
   Containers::Vector< Real, Device > u, v;
   u. setSize( size );
   v. setSize( size );
   setOscilatingSequence( u, 1.0 );
   setOnesSequence( v );

   ASSERT_TRUE( Containers::VectorOperations< Device > :: getScalarProduct( u, v ) == 1.0 );
}

TEST( VectorOperationsTest, addVectorTest )
{
   const int size( 10000 );
   Containers::Vector< Real, Device > x, y;
   x.setSize( size );
   y.setSize( size );
   setLinearSequence( x );
   setOnesSequence( y );
   Containers::VectorOperations< Device >::addVector( y, x, 3.0 );

   for( int i = 0; i < size; i ++ )
      ASSERT_TRUE( y.getElement( i ) == 1.0 + 3.0 * i );
};

TEST( VectorOperationsTest, prefixSumTest )
{
   const int size( 10000 );
   Containers::Vector< Real, Device > v;
   v.setSize( size );

   setOnesSequence( v );
   v.computePrefixSum();
   for( int i = 0; i < size; i++ )
      ASSERT_TRUE( v.getElement( i ) == i + 1 );

   v.setValue( 0 );
   v.computePrefixSum();
   for( int i = 0; i < size; i++ )
      ASSERT_TRUE( v.getElement( i ) == 0 );

   setLinearSequence( v );
   v.computePrefixSum();
   for( int i = 1; i < size; i++ )
      ASSERT_TRUE( v.getElement( i ) - v.getElement( i - 1 ) == i );

};

TEST( VectorOperationsTest, exclusivePrefixSumTest )
{
   const int size( 10000 );
   Containers::Vector< Real, Device > v;
   v.setSize( size );

   setOnesSequence( v );
   v.computeExclusivePrefixSum();
   for( int i = 0; i < size; i++ )
      ASSERT_TRUE( v.getElement( i ) == i );

   v.setValue( 0 );
   v.computeExclusivePrefixSum();
   for( int i = 0; i < size; i++ )
      ASSERT_TRUE( v.getElement( i ) == 0 );

   setLinearSequence( v );
   v.computeExclusivePrefixSum();
   for( int i = 1; i < size; i++ )
      ASSERT_TRUE( v.getElement( i ) - v.getElement( i - 1 ) == i - 1 );

};

#endif /* HAVE_GTEST */

int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   return EXIT_FAILURE;
#endif
}
