/***************************************************************************
                          VectorTest-6.h  -  description
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
constexpr int VECTOR_TEST_SIZE = 500;

TYPED_TEST( VectorTest, horizontalOperations )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   using IndexType = typename VectorType::IndexType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size ), _w( size );
   ViewType u( _u ), v( _v ), w( _w );
   EXPECT_EQ( u.getSize(), size );
   u = 0;
   v = 1;
   w = 2;

   u = u + 4 * TNL::max( v, 0 );
   EXPECT_TRUE( u.containsOnlyValue( 4.0 ) );

   u = u + 3 * w + 4 * TNL::max( v, 0 );
   EXPECT_TRUE( u.containsOnlyValue( 14.0 ) );
}

TYPED_TEST( VectorTest, verticalOperations )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   using IndexType = typename VectorType::IndexType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size ), _w( size );
   ViewType u( _u ), v( _v ), w( _w );
   RealType sum_( 0.0 ), absSum( 0.0 ), diffSum( 0.0 ), diffAbsSum( 0.0 ),
   absMin( size + 10.0 ), absMax( -size - 10.0 ),
   diffMin( 2 * size + 10.0 ), diffMax( - 2.0 * size - 10.0 ),
   l2Norm( 0.0 ), l2NormDiff( 0.0 ), argMinValue( size * size ), argMaxValue( -size * size );
   IndexType argMin( 0 ), argMax( 0 );
   for( int i = 0; i < size; i++ )
   {
      const RealType aux = ( RealType )( i - size / 2 ) / ( RealType ) size;
      const RealType w_value = aux * aux - 5.0;
      u.setElement( i, aux );
      v.setElement( i, -aux );
      w.setElement( i, w_value );
      absMin = TNL::min( absMin, TNL::abs( aux ) );
      absMax = TNL::max( absMax, TNL::abs( aux ) );
      diffMin = TNL::min( diffMin, 2 * aux );
      diffMax = TNL::max( diffMax, 2 * aux );
      sum_ += aux;
      absSum += TNL::abs( aux );
      diffSum += 2.0 * aux;
      diffAbsSum += TNL::abs( 2.0* aux );
      l2Norm += aux * aux;
      l2NormDiff += 4.0 * aux * aux;
      if( w_value < argMinValue ) {
         argMinValue = w_value;
         argMin = i;
      }
      if( w_value > argMaxValue ) {
         argMaxValue = w_value;
         argMax = i;
      }
   }
   l2Norm = TNL::sqrt( l2Norm );
   l2NormDiff = TNL::sqrt( l2NormDiff );

   EXPECT_EQ( min( u ), u.getElement( 0 ) );
   EXPECT_EQ( max( u ), u.getElement( size - 1 ) );
   EXPECT_NEAR( sum( u ), sum_, 2.0e-5 );
   EXPECT_EQ( min( abs( u ) ), absMin );
   EXPECT_EQ( max( abs( u ) ), absMax );
   EXPECT_EQ( min( u - v ), diffMin );
   EXPECT_EQ( max( u - v ), diffMax );
   EXPECT_NEAR( sum( u - v ), diffSum, 2.0e-5 );
   EXPECT_NEAR( sum( abs( u - v ) ), diffAbsSum, 2.0e-5 );
   EXPECT_NEAR( lpNorm( u, 2.0 ), l2Norm, 2.0e-5 );
   EXPECT_NEAR( lpNorm( u - v, 2.0 ), l2NormDiff, 2.0e-5 );
   IndexType wArgMin, wArgMax;
   EXPECT_NEAR( TNL::argMin( w, wArgMin ), argMinValue, 2.0e-5 );
   EXPECT_EQ( argMin, wArgMin );
   EXPECT_NEAR( TNL::argMax( w, wArgMax ), argMaxValue, 2.0e-5 );
   EXPECT_EQ( argMax, wArgMax );
}

TYPED_TEST( VectorTest, scalarProduct )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size );
   ViewType u( _u ), v( _v );
   RealType aux( 0.0 );
   for( int i = 0; i < size; i++ )
   {
      const RealType x = i;
      const RealType y = size / 2 - i;
      u.setElement( i, x );
      v.setElement( i, y );
      aux += x * y;
   }
   EXPECT_NEAR( ( u, v ), aux, 1.0e-5 );
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
