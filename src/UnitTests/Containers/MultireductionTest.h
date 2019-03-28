/***************************************************************************
                          MultireductionTest.h  -  description
                             -------------------
    begin                : Oct 1, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#ifdef HAVE_GTEST
#include "gtest/gtest.h"

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Containers/Algorithms/Multireduction.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Containers::Algorithms;

template< typename View >
void setLinearSequence( View& deviceVector )
{
   using HostVector = Containers::Vector< typename View::RealType, Devices::Host, typename View::IndexType >;
   HostVector a;
   a.setLike( deviceVector );
   for( int i = 0; i < a.getSize(); i++ )
      a[ i ] = i;
   deviceVector = a;
}

template< typename View >
void setNegativeLinearSequence( View& deviceVector )
{
   using HostVector = Containers::Vector< typename View::RealType, Devices::Host, typename View::IndexType >;
   HostVector a;
   a.setLike( deviceVector );
   for( int i = 0; i < a.getSize(); i++ )
      a[ i ] = -i;
   deviceVector = a;
}

// test fixture for typed tests
template< typename Vector >
class MultireductionTest : public ::testing::Test
{
protected:
   using DeviceVector = Vector;
   using DeviceView = VectorView< typename Vector::RealType, typename Vector::DeviceType, typename Vector::IndexType >;
   using HostVector = typename DeviceVector::HostType;
   using HostView = typename DeviceView::HostType;

   // should be small enough to have fast tests, but larger than minGPUReductionDataSize
   // and large enough to require multiple CUDA blocks for reduction
   static constexpr int size = 5000;

   // number of vectors which are reduced together
   static constexpr int n = 4;

   DeviceVector V;
   DeviceVector y;
   HostVector result;

   MultireductionTest()
   {
      V.setSize( size * n );
      y.setSize( size );
      result.setSize( n );

      for( int i = 0; i < n; i++ ) {
         DeviceView v( &V[ i * size ], size );
         if( i % 2 == 0 )
            setLinearSequence( v );
         else
            setNegativeLinearSequence( v );
      }
      y.setValue( 1 );
   }
};

// types for which MultireductionTest is instantiated
using VectorTypes = ::testing::Types<
   Vector< int,   Devices::Host >,
   Vector< float, Devices::Host >
#ifdef HAVE_CUDA
   ,
   Vector< int,   Devices::Cuda >,
   Vector< float, Devices::Cuda >
#endif
>;

TYPED_TEST_SUITE( MultireductionTest, VectorTypes );

TYPED_TEST( MultireductionTest, scalarProduct )
{
   using RealType = typename TestFixture::DeviceVector::RealType;
   using DeviceType = typename TestFixture::DeviceVector::DeviceType;

   ParallelReductionScalarProduct< RealType, RealType > scalarProduct;
   Multireduction< DeviceType >::reduce
               ( scalarProduct,
                 this->n,
                 this->size,
                 this->V.getData(),
                 this->size,
                 this->y.getData(),
                 this->result.getData() );

   for( int i = 0; i < this->n; i++ ) {
      if( i % 2 == 0 )
         EXPECT_EQ( this->result[ i ], 0.5 * this->size * ( this->size - 1 ) );
      else
         EXPECT_EQ( this->result[ i ], - 0.5 * this->size * ( this->size - 1 ) );
   }
}

#endif // HAVE_GTEST


#include "../GtestMissingError.h"
int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   throw GtestMissingError();
#endif
}
