/***************************************************************************
                          VectorTest-5.h  -  description
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

TYPED_TEST( VectorTest, sin )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size );
   ViewType u( _u ), v( _v );
   for( int i = 0; i < size; i++ )
   {
      u.setElement( i, ( RealType ) i - ( RealType ) size / 2 );
      v.setElement( i, TNL::sin( ( RealType ) i - ( RealType ) size / 2 ) );
   }

   //EXPECT_EQ( sin( u ), v );
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( sin( u ).getElement( i ), v.getElement( i ), 1.0e-6 );
}

TYPED_TEST( VectorTest, cos )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size );
   ViewType u( _u ), v( _v );
   for( int i = 0; i < size; i++ )
   {
      u.setElement( i, ( RealType ) i - ( RealType ) size / 2 );
      v.setElement( i, TNL::cos( ( RealType ) i - ( RealType ) size / 2 ) );
   }

   //EXPECT_EQ( cos( u ), v );
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( cos( u ).getElement( i ), v.getElement( i ), 1.0e-6 );
}

TYPED_TEST( VectorTest, tan )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   const int size = VECTOR_TEST_SIZE;
   const double h = 10.0 / size;

   VectorType _u( size ), _v( size );
   ViewType u( _u ), v( _v );
   for( int i = 0; i < size; i++ )
   {
      const RealType x = -5.0 + i * h;
      u.setElement( i, x );
      v.setElement( i, TNL::tan( x ) );
   }

   //EXPECT_EQ( tan( u ), v );
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( tan( u ).getElement( i ), v.getElement( i ), 1.0e-6 );

}

TYPED_TEST( VectorTest, sqrt )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size );
   ViewType u( _u ), v( _v );
   for( int i = 0; i < size; i++ )
   {
      u.setElement( i, ( RealType ) i );
      v.setElement( i, TNL::sqrt( ( RealType ) i ) );
   }

   //EXPECT_EQ( sqrt( u ), v );
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( sqrt( u ).getElement( i ), v.getElement( i ), 1.0e-6 );

}

TYPED_TEST( VectorTest, cbrt )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size );
   ViewType u( _u ), v( _v );
   for( int i = 0; i < size; i++ )
   {
      u.setElement( i, ( RealType ) i );
      v.setElement( i, TNL::cbrt( ( RealType ) i ) );
   }

   //EXPECT_EQ( cbrt( u ), v );
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( cbrt( u ).getElement( i ), v.getElement( i ), 1.0e-6 );

}

TYPED_TEST( VectorTest, pow )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size ), _w( size );
   ViewType u( _u ), v( _v ), w( _w );
   for( int i = 0; i < size; i++ )
   {
      u.setElement( i, ( RealType ) i - ( RealType ) size / 2 );
      v.setElement( i, TNL::pow( ( RealType ) i - ( RealType ) size / 2, 2.0 ) );
      w.setElement( i, TNL::pow( ( RealType ) i - ( RealType ) size / 2, 3.0 ) );
   }

   //EXPECT_EQ( pow( u, 2.0 ), v );
   //EXPECT_EQ( pow( u, 3.0 ), w );
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( pow( u, 2.0 ).getElement( i ), v.getElement( i ), 1.0e-6 );
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( pow( u, 3.0 ).getElement( i ), w.getElement( i ), 1.0e-6 );


}

TYPED_TEST( VectorTest, floor )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size );
   ViewType u( _u ), v( _v );
   for( int i = 0; i < size; i++ )
   {
      u.setElement( i, ( RealType ) i - ( RealType ) size / 2 );
      v.setElement( i, TNL::floor( ( RealType ) i - ( RealType ) size / 2 ) );
   }

   //EXPECT_EQ( floor( u ), v );
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( floor( u ).getElement( i ), v.getElement( i ), 1.0e-6 );

}

TYPED_TEST( VectorTest, ceil )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size );
   ViewType u( _u ), v( _v );
   for( int i = 0; i < size; i++ )
   {
      u.setElement( i, ( RealType ) i - ( RealType ) size / 2 );
      v.setElement( i, TNL::ceil( ( RealType ) i - ( RealType ) size / 2 ) );
   }

   //EXPECT_EQ( ceil( u ), v );
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( ceil( u ).getElement( i ), v.getElement( i ), 1.0e-6 );

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
