/***************************************************************************
                          VectorTest-4.h  -  description
                             -------------------
    begin                : Oct 25, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// NOTE: Vector = Array + VectorOperations, so we test Vector and VectorOperations at the same time

#pragma once

#ifdef HAVE_GTEST
#include <limits>

#include <TNL/Experimental/Arithmetics/Quad.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include "VectorTestSetup.h"

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Containers::Algorithms;
using namespace TNL::Arithmetics;

// should be small enough to have fast tests, but larger than minGPUReductionDataSize
// and large enough to require multiple CUDA blocks for reduction
constexpr int VECTOR_TEST_SIZE = 5000;

TYPED_TEST( VectorTest, addVector )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   const int size = VECTOR_TEST_SIZE;

   VectorType x, y;
   x.setSize( size );
   y.setSize( size );
   ViewType x_view( x ), y_view( y );

   typename VectorType::HostType expected1, expected2;
   expected1.setSize( size );
   expected2.setSize( size );
   for( int i = 0; i < size; i++ ) {
      expected1[ i ] = 2.0 + 3.0 * i;
      expected2[ i ] = 1.0 + 3.0 * i;
   }

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   VectorOperations::addVector( x, y, 3.0, 2.0 );
   EXPECT_EQ( x, expected1 );

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   x.addVector( y, 3.0, 1.0 );
   EXPECT_EQ( x, expected2 );

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   x_view.addVector( y_view, 3.0, 1.0 );
   EXPECT_EQ( x, expected2 );

   // multiplication by floating-point scalars which produces integer values
   setConstantSequence( x, 2 );
   setConstantSequence( y, 4 );
   x.addVector( y, 2.5, -1.5 );
   EXPECT_EQ( x.min(), 7 );
   EXPECT_EQ( x.max(), 7 );
}

TYPED_TEST( VectorTest, addVectors )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   const int size = VECTOR_TEST_SIZE;

   VectorType x, y, z;
   x.setSize( size );
   y.setSize( size );
   z.setSize( size );
   ViewType x_view( x ), y_view( y ), z_view( z );

   typename VectorType::HostType expected1, expected2;
   expected1.setSize( size );
   expected2.setSize( size );
   for( int i = 0; i < size; i++ ) {
      expected1[ i ] = 1.0 + 3.0 * i + 2.0;
      expected2[ i ] = 2.0 + 3.0 * i + 2.0;
   }

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   setConstantSequence( z, 2 );
   VectorOperations::addVectors( x, y, 3.0, z, 1.0, 1.0 );
   EXPECT_EQ( x, expected1 );

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   setConstantSequence( z, 2 );
   x.addVectors( y, 3.0, z, 1.0, 2.0 );
   EXPECT_EQ( x, expected2 );

   setConstantSequence( x, 1 );
   setLinearSequence( y );
   setConstantSequence( z, 2 );
   x_view.addVectors( y_view, 3.0, z_view, 1.0, 2.0 );
   EXPECT_EQ( x, expected2 );

   // multiplication by floating-point scalars which produces integer values
   setConstantSequence( x, 2 );
   setConstantSequence( y, 4 );
   setConstantSequence( z, 6 );
   x.addVectors( y, 2.5, z, -1.5, -1.5 );
   EXPECT_EQ( x.min(), -2 );
   EXPECT_EQ( x.max(), -2 );
}

TYPED_TEST( VectorTest, prefixSum )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   using DeviceType = typename VectorType::DeviceType;
   using IndexType = typename VectorType::IndexType;
   const int size = VECTOR_TEST_SIZE;

   VectorType v( size );
   ViewType v_view( v );

   v = 0;
   v.computePrefixSum();
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v.getElement( i ), 0 );

   setLinearSequence( v );
   v.computePrefixSum();
   for( int i = 1; i < size; i++ )
      EXPECT_EQ( v.getElement( i ) - v.getElement( i - 1 ), i );

   setConstantSequence( v, 1 );
   v_view.computePrefixSum();
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v.getElement( i ), i + 1 );

   v = 0;
   v_view.computePrefixSum();
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v.getElement( i ), 0 );

   setLinearSequence( v );
   v_view.computePrefixSum();
   for( int i = 1; i < size; i++ )
      EXPECT_EQ( v.getElement( i ) - v.getElement( i - 1 ), i );
}

TYPED_TEST( VectorTest, exclusivePrefixSum )
{
   using VectorType = typename TestFixture::VectorType;
   using VectorOperations = typename TestFixture::VectorOperations;
   using ViewType = typename TestFixture::ViewType;
   const int size = VECTOR_TEST_SIZE;

   VectorType v;
   v.setSize( size );
   ViewType v_view( v );

   setConstantSequence( v, 1 );
   v.computeExclusivePrefixSum();
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v.getElement( i ), i );

   v.setValue( 0 );
   v.computeExclusivePrefixSum();
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v.getElement( i ), 0 );

   setLinearSequence( v );
   v.computeExclusivePrefixSum();
   for( int i = 1; i < size; i++ )
      EXPECT_EQ( v.getElement( i ) - v.getElement( i - 1 ), i - 1 );

   setConstantSequence( v, 1 );
   v_view.computeExclusivePrefixSum();
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v.getElement( i ), i );

   v.setValue( 0 );
   v_view.computeExclusivePrefixSum();
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v.getElement( i ), 0 );

   setLinearSequence( v );
   v_view.computeExclusivePrefixSum();
   for( int i = 1; i < size; i++ )
      EXPECT_EQ( v.getElement( i ) - v.getElement( i - 1 ), i - 1 );
}

TYPED_TEST( VectorTest, segmentedPrefixSum )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   using DeviceType = typename VectorType::DeviceType;
   using IndexType = typename VectorType::IndexType;
   using FlagsArrayType = Array< bool, DeviceType, IndexType >;
   using FlagsViewType = ArrayView< bool, DeviceType, IndexType >;
   const int size = VECTOR_TEST_SIZE;

   VectorType v( size );
   ViewType v_view( v );

   FlagsArrayType flags( size ), flags_copy( size );
   FlagsViewType flags_view( flags );
   flags_view.evaluate( [] __cuda_callable__ ( IndexType i ) { return ( i % 5 ) == 0; } );
   flags_copy = flags_view;

   v = 0;
   v.computeSegmentedPrefixSum( flags_view );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v.getElement( i ), 0 );
   flags_view = flags_copy;
   
   v = 1;
   v.computeSegmentedPrefixSum( flags_view );
   for( int i = 0; i < size; i++ )
         EXPECT_EQ( v.getElement( i ), ( i % 5 ) + 1 );
   flags_view = flags_copy;

   setLinearSequence( v );
   v.computeSegmentedPrefixSum( flags_view );
   for( int i = 1; i < size; i++ )
   {
      if( flags.getElement( i ) )
         EXPECT_EQ( v.getElement( i ), i );
      else
         EXPECT_EQ( v.getElement( i ) - v.getElement( i - 1 ), i );
   }
   flags_view = flags_copy;
   
   v_view = 0;
   v_view.computeSegmentedPrefixSum( flags_view );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_view.getElement( i ), 0 );
   flags_view = flags_copy;
   
   v_view = 1;
   v_view.computeSegmentedPrefixSum( flags_view );
   for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_view.getElement( i ), ( i % 5 ) + 1 );
   flags_view = flags_copy;

   v_view.evaluate( [] __cuda_callable__ ( IndexType i ) { return i; } );
   v_view.computeSegmentedPrefixSum( flags_view );
   for( int i = 1; i < size; i++ )
   {
      if( flags.getElement( i ) )
         EXPECT_EQ( v_view.getElement( i ), i );
      else
         EXPECT_EQ( v_view.getElement( i ) - v_view.getElement( i - 1 ), i );
   }
}


TYPED_TEST( VectorTest, abs )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size );
   ViewType u( _u ), v( _v );
   for( int i = 0; i < size; i++ )
      u.setElement( i, i );

   v = -u;
   EXPECT_TRUE( abs( v ) == u );
}

#endif // HAVE_GTEST


#include "../GtestMissingError.h"
int main( int argc, char* argv[] )
{
   //Test();
   //return 0;
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   throw GtestMissingError();
#endif
}
