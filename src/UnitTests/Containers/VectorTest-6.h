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
constexpr int VECTOR_TEST_SIZE = 500;

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

   //EXPECT_EQ( sign( u ), v );
   for( int i = 0; i < size; i++ )
      EXPECT_NEAR( sign( u ).getElement( i ), sign( u.getElement( i ) ), 1.0e-6 );
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

   const int sum = TNL::sum( v );
   const int l1norm = v.lpNorm< int >( 1.0 );
   const float l2norm = v.lpNorm< float >( 2.0 );
   const float l3norm = v.lpNorm< float >( 3.0 );
   EXPECT_EQ( sum, 512 );
   EXPECT_EQ( l1norm, 512 );
   EXPECT_NEAR( l2norm, std::sqrt( 512 ), epsilon );
   EXPECT_NEAR( l3norm, std::cbrt( 512 ), epsilon );
   
   const int diff_sum = TNL::sum( v - w ); //v.differenceSum< int >( w );
   const int diff_l1norm = lpNorm( v - w, 1.0 );//v.differenceLpNorm< int >( w, 1.0 );
   const float diff_l2norm = lpNorm( v - w, 2.0 );//v.differenceLpNorm< float >( w, 2.0 );
   const float diff_l3norm = lpNorm( v - w, 3.0 );//v.differenceLpNorm< float >( w, 3.0 );
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

   const int diff_sum_view = TNL::sum( v_view - w_view );
   const int diff_l1norm_view = lpNorm( v_view - w_view, 1.0 );
   const float diff_l2norm_view = lpNorm( v_view - w_view, 2.0 );
   const float diff_l3norm_view = lpNorm( v_view - w_view, 3.0 );
   EXPECT_EQ( diff_sum_view, 512 );
   EXPECT_EQ( diff_l1norm_view, 512 );
   EXPECT_NEAR( diff_l2norm_view, std::sqrt( 512 ), epsilon );
   EXPECT_NEAR( diff_l3norm_view, std::cbrt( 512 ), epsilon );
}

TEST( VectorSpecialCasesTest, assignmentThroughView )
{
   using VectorType = Containers::Vector< int, Devices::Host >;
   using ViewType = VectorView< int, Devices::Host >;

   static_assert( Containers::Algorithms::detail::HasSubscriptOperator< VectorType >::value, "Subscript operator detection by SFINAE does not work for Vector." );
   static_assert( Containers::Algorithms::detail::HasSubscriptOperator< ViewType >::value, "Subscript operator detection by SFINAE does not work for VectorView." );

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

   EXPECT_EQ( max( u_view ), 1 );
   EXPECT_EQ( min( u_view ), 1 );
   EXPECT_EQ( max( abs( u_view ) ), 1 );
   EXPECT_EQ( min( abs( u_view ) ), 1 );
   EXPECT_EQ( lpNorm( u_view, 1 ), 100 );
   EXPECT_EQ( max( u_view - v_view ), 0 );
   EXPECT_EQ( min( u_view - v_view ), 0 );
   EXPECT_EQ( max( abs( u_view - v_view ) ), 0 );
   EXPECT_EQ( min( abs( u_view - v_view ) ), 0 );
   EXPECT_EQ( lpNorm( u_view - v_view, 1.0 ), 0 );
   EXPECT_EQ( sum( u_view - v_view ), 0 );
   EXPECT_EQ( dot( u_view, v_view ), 100 );
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

#include "../main.h"
