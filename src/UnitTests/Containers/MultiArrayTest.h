/***************************************************************************
                          MultiArrayTester.h -  description
                             -------------------
    begin                : Jul 4, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/MultiArray.h>

#ifdef HAVE_GTEST 
#include "gtest/gtest.h"
#endif

using namespace TNL;
using namespace TNL::Containers;

#ifdef HAVE_CUDA
template< typename ValueType, typename IndexType >
__global__ void testSetGetElementKernel( MultiArray< 1, ValueType, Devices::Cuda, IndexType >* u )
{
   if( threadIdx.x < ( *u ).getDimensions().x() )
      ( *u )( threadIdx.x ) = threadIdx.x;
}

template< typename ValueType, typename IndexType >
__global__ void testSetGetElementKernel( MultiArray< 2, ValueType, Devices::Cuda, IndexType >* u )
{
   if( threadIdx.x < ( *u ).getDimensions().x() &&
       threadIdx.x < ( *u ).getDimensions().y() )
      ( *u )( threadIdx.x, threadIdx.x ) = threadIdx.x;
}

template< typename ValueType, typename IndexType >
__global__ void testSetGetElementKernel( MultiArray< 3, ValueType, Devices::Cuda, IndexType >* u )
{
   if( threadIdx.x < ( *u ).getDimensions().x() &&
       threadIdx.x < ( *u ).getDimensions().y() &&
       threadIdx.x < ( *u ).getDimensions().z() )
      ( *u )( threadIdx.x, threadIdx.x, threadIdx.x ) = threadIdx.x;
}

#endif /* HAVE_CUDA */

#ifdef HAVE_GTEST

TEST( MultiArrayTest, testConstructorDestructor )
{
   using namespace TNL::Containers;
   MultiArray< Dimension, ValueType, Device, IndexType > u;
}

TEST( MultiArrayTest, testSetSize )
{
   using namespace TNL::Containers;
   MultiArray< Dimension, ValueType, Device, IndexType > u, v;
   u. setDimensions( 10 );
   v. setDimensions( 10 );
}

void setDiagonalElement( Containers::MultiArray< 1, ValueType, Device, IndexType >& u,
                         const IndexType& i,
                         const ValueType& v )
{
   u.setElement( i, v );
}

void setDiagonalElement( Containers::MultiArray< 2, ValueType, Device, IndexType >& u,
                         const IndexType& i,
                         const ValueType& v )
{
   u.setElement( i, i, v );
}

void setDiagonalElement( Containers::MultiArray< 3, ValueType, Device, IndexType >& u,
                         const IndexType& i,
                         const ValueType& v )
{
   u.setElement( i, i, i, v );
}

IndexType getDiagonalElement( Containers::MultiArray< 1, ValueType, Device, IndexType >& u,
                              const IndexType& i )
{
   return u.getElement( i );
}

IndexType getDiagonalElement( Containers::MultiArray< 2, ValueType, Device, IndexType >& u,
                              const IndexType& i )
{
   return u.getElement( i, i );
}

IndexType getDiagonalElement( Containers::MultiArray< 3, ValueType, Device, IndexType >& u,
                              const IndexType& i )
{
   return u.getElement( i, i, i );
}


TEST( MultiArrayTest, testSetGetElement )
{
   using namespace TNL::Containers;
   MultiArray< Dimension, ValueType, Device, IndexType > u;
   u. setDimensions( 10 );
   if( std::is_same< Device, Devices::Host >::value )
   {
      for( int i = 0; i < 10; i ++ )
         this->setDiagonalElement( u, i, i  );
   }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      MultiArray< Dimension, ValueType, Device, IndexType >* kernel_u =
               Devices::Cuda::passToDevice( u );
      testSetGetElementKernel<<< 1, 16 >>>( kernel_u );
      Devices::Cuda::freeFromDevice( kernel_u );
      ASSERT_TRUE( TNL_CHECK_CUDA_DEVICE );
#endif
   }
   for( int i = 0; i < 10; i ++ )
      ASSERT_EQ( getDiagonalElement( u, i ), i );
};

TEST( MultiArrayTest, testComparisonOperator )
{
   using namespace TNL::Containers;
   MultiArray< Dimension, ValueType, Device, IndexType > u, v, w;
   u.setDimensions( 10 );
   v.setDimensions( 10 );
   w.setDimensions( 10 );
   u.setValue( 0 );
   v.setValue( 0 );
   w.setValue( 0 );
   for( int i = 0; i < 10; i ++ )
   {
      setDiagonalElement( u, i, i );
      setDiagonalElement( v, i, i );
      setDiagonalElement( w, i, 2*1 );
   }
   ASSERT_TRUE( u == v );
   ASSERT_FALSE( u != v );
   ASSERT_TRUE( u != w );
   ASSERT_FALSE( u == w );
};

TEST( MultiArrayTest, testEquivalenceOperator )
{
   using namespace TNL::Containers;
   MultiArray< Dimension, ValueType, Device, IndexType > u;
   MultiArray< Dimension, ValueType, Device, IndexType > v;
   u. setDimensions( 10 );
   v. setDimensions( 10 );
   for( int i = 0; i < 10; i ++ )
      setDiagonalElement( u, i, i );
   v = u;
   ASSERT_TRUE( u == v );
   ASSERT_FALSE( u != v );
};

TEST( MultiArrayTest, testGetSize )
{
   using namespace TNL::Containers;
   MultiArray< Dimension, ValueType, Device, IndexType > u;
   const int maxSize = 10;
   for( int i = 1; i < maxSize; i ++ )
      u. setDimensions( i );

   ASSERT_EQ( u. getDimensions().x(), maxSize - 1 );
};

TEST( MultiArrayTest, testReset )
{
   using namespace TNL::Containers;
   MultiArray< Dimension, ValueType, Device, IndexType > u;
   u.setDimensions( 100 );
   ASSERT_EQ( u. getDimensions().x(), 100 );
   u.reset();
   ASSERT_EQ( u. getDimensions().x(), 0 );
   u.setDimensions( 100 );
   ASSERT_EQ( u. getDimensions().x(), 100 );
   u.reset();
   ASSERT_EQ( u. getDimensions().x(), 0 );

};

TEST( MultiArrayTest, testSetSizeAndDestructor )
{
   using namespace TNL::Containers;
   for( int i = 1; i < 100; i ++ )
   {
      MultiArray< Dimension, ValueType, Device, IndexType > u;
      u. setDimensions( i );
   }
}

TEST( MultiArrayTest, testSaveAndLoad )
{
   using namespace TNL::Containers;
   MultiArray< Dimension, ValueType, Device, IndexType > v;
   const int size( 10 );
   ASSERT_TRUE( v. setDimensions( size ) );
   for( int i = 0; i < size; i ++ )
      setDiagonalElement( v, i, 3.14147 );
   File file;
   file. open( "test-file.tnl", IOMode::write );
   ASSERT_TRUE( v. save( file ) );
   file. close();
   MultiArray< Dimension, ValueType, Device, IndexType > u;
   file. open( "test-file.tnl", IOMode::read );
   ASSERT_TRUE( u. load( file ) );
   file. close();
   ASSERT_TRUE( u == v );

   EXPECT_EQ( std::remove( "test-file.tnl" ), 0 );
}
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






