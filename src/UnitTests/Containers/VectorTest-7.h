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
#include "VectorTestSetup.h"

// should be small enough to have fast tests, but larger than minGPUReductionDataSize
// and large enough to require multiple CUDA blocks for reduction
constexpr int VECTOR_TEST_SIZE = 100;

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

// NOTE: The following lambdas cannot be inside the test because of nvcc ( v. 10.1.105 )
// error #3049-D: The enclosing parent function ("TestBody") for an extended __host__ __device__ lambda cannot have private or protected access within its class
template< typename VectorView >
typename VectorView::RealType
performEvaluateAndReduce( VectorView& u, VectorView& v, VectorView& w )
{
   using RealType = typename VectorView::RealType;

   auto reduction = [] __cuda_callable__ ( RealType& a, const RealType& b ) { a += b; };
   auto volatileReduction = [] __cuda_callable__ ( volatile RealType& a, volatile RealType& b ) { a += b; };
   return evaluateAndReduce( w, u * v, reduction, volatileReduction, ( RealType ) 0.0 );
}

TYPED_TEST( VectorTest, evaluateAndReduce )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   using IndexType = typename VectorType::IndexType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size ), _w( size );
   ViewType u( _u ), v( _v ), w( _w );
   RealType aux( 0.0 );
   for( int i = 0; i < size; i++ )
   {
      const RealType x = i;
      const RealType y = size / 2 - i;
      u.setElement( i, x );
      v.setElement( i, y );
      aux += x * y;
   }
   auto r = performEvaluateAndReduce( u, v, w );
   EXPECT_TRUE( w == u * v );
   EXPECT_NEAR( aux, r, 1.0e-5 );
}


#endif // HAVE_GTEST

#include "../main.h"
