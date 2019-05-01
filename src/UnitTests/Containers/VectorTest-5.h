/***************************************************************************
                          VectorTest.h  -  description
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

TYPED_TEST( VectorTest, acos )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size );
   ViewType u( _u ), v( _v );
   for( int i = 0; i < size; i++ )
   {
      u.setElement( i, ( RealType )( i - size / 2 ) / ( RealType ) size );
      v.setElement( i, TNL::acos( ( RealType )( i - size / 2 ) / ( RealType ) size ) );
   }

   //EXPECT_EQ( acos( u ), v );
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( acos( u ).getElement( i ), v.getElement( i ), 1.0e-6 );
}

TYPED_TEST( VectorTest, asin )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size );
   ViewType u( _u ), v( _v );
   for( int i = 0; i < size; i++ )
   {
      u.setElement( i, ( RealType ) ( i - size / 2 ) / ( RealType ) size );
      v.setElement( i, TNL::asin( ( RealType )( i - size / 2 ) / ( RealType ) size ) );
   }

   //EXPECT_EQ( asin( u ), v );
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( asin( u ).getElement( i ), v.getElement( i ), 1.0e-6 );
}

TYPED_TEST( VectorTest, atan )
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
      v.setElement( i, TNL::atan( ( RealType ) i - ( RealType ) size / 2 ) );
   }

   //EXPECT_EQ( atan( u ), v );
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( atan( u ).getElement( i ), v.getElement( i ), 1.0e-6 );
}

TYPED_TEST( VectorTest, cosh )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   const int size = VECTOR_TEST_SIZE;

   RealType h = 2.0 / ( RealType ) size;
   VectorType _u( size ), _v( size );
   ViewType u( _u ), v( _v );
   for( int i = 0; i < size; i++ )
   {
      u.setElement( i, i * h - ( RealType ) 1.0 );
      v.setElement( i, TNL::cosh( i * h - ( RealType ) 1.0 ) );
   }

   // EXPECT_EQ( cosh( u ), v ) does not work here for float, maybe because
   // of some fast-math optimization
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( cosh( u ).getElement( i ), v.getElement( i ), 1.0e-6 );
}

TYPED_TEST( VectorTest, tanh )
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
      v.setElement( i, TNL::tanh( ( RealType ) i - ( RealType ) size / 2 ) );
   }

   //EXPECT_EQ( tanh( u ), v );
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( tanh( u ).getElement( i ), v.getElement( i ), 1.0e-6 );
}

TYPED_TEST( VectorTest, log )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size );
   ViewType u( _u ), v( _v );
   for( int i = 0; i < size; i++ )
   {
      u.setElement( i, ( RealType ) i + 1 );
      v.setElement( i, TNL::log( ( RealType ) i + 1 ) );
   }

   //EXPECT_EQ( log( u ), v );
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( log( u ).getElement( i ), v.getElement( i ), 1.0e-6 );

}

TYPED_TEST( VectorTest, log10 )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size );
   ViewType u( _u ), v( _v );
   for( int i = 0; i < size; i++ )
   {
      u.setElement( i, ( RealType ) i + 1 );
      v.setElement( i, TNL::log10( ( RealType ) i + 1 ) );
   }

   // EXPECT_EQ( log10( u ), v ) does not work here for float, maybe because
   // of some fast-math optimization
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( log10( u ).getElement( i ), v.getElement( i ), 1.0e-6 );
}

TYPED_TEST( VectorTest, log2 )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size );
   ViewType u( _u ), v( _v );
   for( int i = 0; i < size; i++ )
   {
      u.setElement( i, ( RealType ) i + 1 );
      v.setElement( i, TNL::log2( ( RealType ) i + 1 ) );
   }

   //EXPECT_EQ( log2( u ), v );
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( log2( u ).getElement( i ), v.getElement( i ), 1.0e-6 );

}

TYPED_TEST( VectorTest, exp )
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
      u.setElement( i, x  );
      v.setElement( i, TNL::exp( x ) );
   }

   //EXPECT_EQ( exp( u ), v );
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( exp( u ).getElement( i ), v.getElement( i ), 1.0e-6 );
}

TYPED_TEST( VectorTest, sign )
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
      v.setElement( i, TNL::sign( ( RealType ) i - ( RealType ) size / 2 ) );
   }

   //EXPECT_EQ( sign( u ), v );
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( sign( u ).getElement( i ), v.getElement( i ), 1.0e-6 );
}

// TODO: test prefix sum with custom begin and end parameters

TEST( VectorSpecialCasesTest, sumOfBoolVector )
{
   using VectorType = Containers::Vector< bool, Devices::Host >;
   using ViewType = VectorView< bool, Devices::Host >;
   const float epsilon = 64 * std::numeric_limits< float >::epsilon();

   VectorType v( 512 ), w( 512 );
   ViewType v_view( v ), w_view( w );
   v.setValue( true );
   w.setValue( false );

   const int sum = v.sum< int >();
   const int l1norm = v.lpNorm< int >( 1.0 );
   const float l2norm = v.lpNorm< float >( 2.0 );
   const float l3norm = v.lpNorm< float >( 3.0 );
   EXPECT_EQ( sum, 512 );
   EXPECT_EQ( l1norm, 512 );
   EXPECT_NEAR( l2norm, std::sqrt( 512 ), epsilon );
   EXPECT_NEAR( l3norm, std::cbrt( 512 ), epsilon );

   const int diff_sum = v.differenceSum< int >( w );
   const int diff_l1norm = v.differenceLpNorm< int >( w, 1.0 );
   const float diff_l2norm = v.differenceLpNorm< float >( w, 2.0 );
   const float diff_l3norm = v.differenceLpNorm< float >( w, 3.0 );
   EXPECT_EQ( diff_sum, 512 );
   EXPECT_EQ( diff_l1norm, 512 );
   EXPECT_NEAR( diff_l2norm, std::sqrt( 512 ), epsilon );
   EXPECT_NEAR( diff_l3norm, std::cbrt( 512 ), epsilon );

   // test views
   const int sum_view = v_view.sum< int >();
   const int l1norm_view = v_view.lpNorm< int >( 1.0 );
   const float l2norm_view = v_view.lpNorm< float >( 2.0 );
   const float l3norm_view = v_view.lpNorm< float >( 3.0 );
   EXPECT_EQ( sum_view, 512 );
   EXPECT_EQ( l1norm_view, 512 );
   EXPECT_NEAR( l2norm_view, std::sqrt( 512 ), epsilon );
   EXPECT_NEAR( l3norm_view, std::cbrt( 512 ), epsilon );

   const int diff_sum_view = v_view.differenceSum< int >( w_view );
   const int diff_l1norm_view = v_view.differenceLpNorm< int >( w_view, 1.0 );
   const float diff_l2norm_view = v_view.differenceLpNorm< float >( w_view, 2.0 );
   const float diff_l3norm_view = v_view.differenceLpNorm< float >( w_view, 3.0 );
   EXPECT_EQ( diff_sum_view, 512 );
   EXPECT_EQ( diff_l1norm_view, 512 );
   EXPECT_NEAR( diff_l2norm_view, std::sqrt( 512 ), epsilon );
   EXPECT_NEAR( diff_l3norm_view, std::cbrt( 512 ), epsilon );
}

TEST( VectorSpecialCasesTest, assignmentThroughView )
{
   using VectorType = Containers::Vector< int, Devices::Host >;
   using ViewType = VectorView< int, Devices::Host >;

   static_assert( Algorithms::Details::HasSubscriptOperator< VectorType >::value, "Subscript operator detection by SFINAE does not work for Vector." );
   static_assert( Algorithms::Details::HasSubscriptOperator< ViewType >::value, "Subscript operator detection by SFINAE does not work for VectorView." );

   VectorType u( 100 ), v( 100 );
   ViewType u_view( u ), v_view( v );

   u.setValue( 42 );
   v.setValue( 0 );
   v_view = u_view;
   EXPECT_EQ( u_view.getData(), u.getData() );
   EXPECT_EQ( v_view.getData(), v.getData() );
   for( int i = 0; i < 100; i++ )
      EXPECT_EQ( v_view[ i ], 42 );

   u.setValue( 42 );
   v.setValue( 0 );
   v_view = u;
   EXPECT_EQ( u_view.getData(), u.getData() );
   EXPECT_EQ( v_view.getData(), v.getData() );
   for( int i = 0; i < 100; i++ )
      EXPECT_EQ( v_view[ i ], 42 );
}

TEST( VectorSpecialCasesTest, operationsOnConstView )
{
   using VectorType = Containers::Vector< int, Devices::Host >;
   using ViewType = VectorView< const int, Devices::Host >;

   VectorType u( 100 ), v( 100 );
   ViewType u_view( u ), v_view( v );

   u.setValue( 1 );
   v.setValue( 1 );

   EXPECT_EQ( u_view.max(), 1 );
   EXPECT_EQ( u_view.min(), 1 );
   EXPECT_EQ( u_view.absMax(), 1 );
   EXPECT_EQ( u_view.absMin(), 1 );
   EXPECT_EQ( u_view.lpNorm( 1 ), 100 );
   EXPECT_EQ( u_view.differenceMax( v_view ), 0 );
   EXPECT_EQ( u_view.differenceMin( v_view ), 0 );
   EXPECT_EQ( u_view.differenceAbsMax( v_view ), 0 );
   EXPECT_EQ( u_view.differenceAbsMin( v_view ), 0 );
   EXPECT_EQ( u_view.differenceLpNorm( v_view, 1 ), 0 );
   EXPECT_EQ( u_view.differenceSum( v_view ), 0 );
   EXPECT_EQ( u_view.scalarProduct( v_view ), 100 );
}

TEST( VectorSpecialCasesTest, initializationOfVectorViewByArrayView )
{
   using ArrayType = Containers::Array< int, Devices::Host >;
   using VectorViewType = VectorView< const int, Devices::Host >;
   using ArrayViewType = ArrayView< int, Devices::Host >;

   ArrayType a( 100 );
   a.setValue( 0 );
   ArrayViewType a_view( a );

   VectorViewType v_view( a_view );
   EXPECT_EQ( v_view.getData(), a_view.getData() );
   EXPECT_EQ( v_view.sum(), 0 );
}

TEST( VectorSpecialCasesTest, defaultConstructors )
{
   using ArrayType = Containers::Array< int, Devices::Host >;
   using VectorViewType = VectorView< int, Devices::Host >;
   using ArrayViewType = ArrayView< int, Devices::Host >;

   ArrayType a( 100 );
   a.setValue( 0 );

   ArrayViewType a_view;
   a_view.bind( a );

   VectorViewType v_view;
   v_view.bind( a );
   EXPECT_EQ( v_view.getData(), a_view.getData() );
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
