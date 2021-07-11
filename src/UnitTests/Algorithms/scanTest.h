#pragma once

#ifdef HAVE_GTEST

#include <TNL/Arithmetics/Quad.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/scan.h>

#include "../Containers/VectorHelperFunctions.h"

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Arithmetics;
using namespace TNL::Algorithms;

// should be small enough to have fast tests, but larger than minGPUReductionDataSize
// and large enough to require multiple CUDA blocks for reduction
constexpr int ARRAY_TEST_SIZE = 10000;

// test fixture for typed tests
template< typename Array >
class ScanTest : public ::testing::Test
{
protected:
   using ArrayType = Array;
   using ViewType = ArrayView< typename Array::ValueType, typename Array::DeviceType, typename Array::IndexType >;
};

// types for which ScanTest is instantiated
// TODO: Quad must be fixed
using ArrayTypes = ::testing::Types<
#ifndef HAVE_CUDA
   Array< int,            Devices::Sequential, short >,
   Array< long,           Devices::Sequential, short >,
   Array< double,         Devices::Sequential, short >,
   //Array< Quad< float >,  Devices::Sequential, short >,
   //Array< Quad< double >, Devices::Sequential, short >,
   Array< int,            Devices::Sequential, int >,
   Array< long,           Devices::Sequential, int >,
   Array< double,         Devices::Sequential, int >,
   //Array< Quad< float >,  Devices::Sequential, int >,
   //Array< Quad< double >, Devices::Sequential, int >,
   Array< int,            Devices::Sequential, long >,
   Array< long,           Devices::Sequential, long >,
   Array< double,         Devices::Sequential, long >,
   //Array< Quad< float >,  Devices::Sequential, long >,
   //Array< Quad< double >, Devices::Sequential, long >,

   Array< int,            Devices::Host, short >,
   Array< long,           Devices::Host, short >,
   Array< double,         Devices::Host, short >,
   //Array< Quad< float >,  Devices::Host, short >,
   //Array< Quad< double >, Devices::Host, short >,
   Array< int,            Devices::Host, int >,
   Array< long,           Devices::Host, int >,
   Array< double,         Devices::Host, int >,
   //Array< Quad< float >,  Devices::Host, int >,
   //Array< Quad< double >, Devices::Host, int >,
   Array< int,            Devices::Host, long >,
   Array< long,           Devices::Host, long >,
   Array< double,         Devices::Host, long >
   //Array< Quad< float >,  Devices::Host, long >,
   //Array< Quad< double >, Devices::Host, long >
#endif
#ifdef HAVE_CUDA
   Array< int,            Devices::Cuda, short >,
   Array< long,           Devices::Cuda, short >,
   Array< double,         Devices::Cuda, short >,
   //Array< Quad< float >,  Devices::Cuda, short >,
   //Array< Quad< double >, Devices::Cuda, short >,
   Array< int,            Devices::Cuda, int >,
   Array< long,           Devices::Cuda, int >,
   Array< double,         Devices::Cuda, int >,
   //Array< Quad< float >,  Devices::Cuda, int >,
   //Array< Quad< double >, Devices::Cuda, int >,
   Array< int,            Devices::Cuda, long >,
   Array< long,           Devices::Cuda, long >,
   Array< double,         Devices::Cuda, long >
   //Array< Quad< float >,  Devices::Cuda, long >,
   //Array< Quad< double >, Devices::Cuda, long >
#endif
>;

TYPED_TEST_SUITE( ScanTest, ArrayTypes );

TYPED_TEST( ScanTest, inclusiveScan )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;
   using ValueType = typename ArrayType::ValueType;
   using DeviceType = typename ArrayType::DeviceType;
   using IndexType = typename ArrayType::IndexType;
   using HostArrayType = typename ArrayType::template Self< ValueType, Devices::Sequential >;
   const int size = ARRAY_TEST_SIZE;

   ArrayType v( size );
   ViewType v_view( v );
   HostArrayType v_host( size );

   setConstantSequence( v, 0 );
   v_host = -1;
   inplaceInclusiveScan( v, 0, size, std::plus<>{}, (ValueType) 0 );
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host = -1;
   inplaceInclusiveScan( v, 0, size, std::plus<>{}, (ValueType) 0 );
   v_host = v_view;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], i + 1 ) << "i = " << i;

   setLinearSequence( v );
   v_host = -1;
   inplaceInclusiveScan( v, 0, size, std::plus<>{}, (ValueType) 0 );
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

   // test views
   setConstantSequence( v, 0 );
   v_host = -1;
   inplaceInclusiveScan( v_view, 0, size, std::plus<>{}, (ValueType) 0 );
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host = -1;
   inplaceInclusiveScan( v_view, 0, size, std::plus<>{}, (ValueType) 0 );
   v_host = v_view;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], i + 1 ) << "i = " << i;

   setLinearSequence( v );
   v_host = -1;
   inplaceInclusiveScan( v_view, 0, size, std::plus<>{}, (ValueType) 0 );
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

   ////
   // With CUDA, perform tests with multiple CUDA grids.
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      Algorithms::detail::CudaScanKernelLauncher< Algorithms::detail::ScanType::Inclusive, ValueType, IndexType >::maxGridSize() = 3;

      setConstantSequence( v, 0 );
      v_host = -1;
      inplaceInclusiveScan( v, 0, size, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::detail::ScanType::Inclusive, ValueType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

      setConstantSequence( v, 1 );
      v_host = -1;
      inplaceInclusiveScan( v, 0, size, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::detail::ScanType::Inclusive, ValueType, IndexType >::gridsCount() ), 1  );
      v_host = v_view;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], i + 1 ) << "i = " << i;

      setLinearSequence( v );
      v_host = -1;
      inplaceInclusiveScan( v, 0, size, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::detail::ScanType::Inclusive, ValueType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

      // test views
      setConstantSequence( v, 0 );
      v_host = -1;
      inplaceInclusiveScan( v_view, 0, size, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::detail::ScanType::Inclusive, ValueType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

      setConstantSequence( v, 1 );
      v_host = -1;
      inplaceInclusiveScan( v_view, 0, size, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::detail::ScanType::Inclusive, ValueType, IndexType >::gridsCount() ), 1  );
      v_host = v_view;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], i + 1 ) << "i = " << i;

      setLinearSequence( v );
      v_host = -1;
      inplaceInclusiveScan( v_view, 0, size, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::detail::ScanType::Inclusive, ValueType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

      Algorithms::detail::CudaScanKernelLauncher< Algorithms::detail::ScanType::Inclusive, ValueType, IndexType >::resetMaxGridSize();
#endif
   }
}

TYPED_TEST( ScanTest, exclusiveScan )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;
   using ValueType = typename ArrayType::ValueType;
   using DeviceType = typename ArrayType::DeviceType;
   using IndexType = typename ArrayType::IndexType;
   using HostArrayType = typename ArrayType::template Self< ValueType, Devices::Sequential >;
   const int size = ARRAY_TEST_SIZE;

   ArrayType v;
   v.setSize( size );
   ViewType v_view( v );
   HostArrayType v_host( size );

   setConstantSequence( v, 0 );
   v_host = -1;
   inplaceExclusiveScan( v, 0, size, std::plus<>{}, (ValueType) 0 );
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host = -1;
   inplaceExclusiveScan( v, 0, size, std::plus<>{}, (ValueType) 0 );
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], i ) << "i = " << i;

   setLinearSequence( v );
   v_host = -1;
   inplaceExclusiveScan( v, 0, size, std::plus<>{}, (ValueType) 0 );
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

   // test views
   setConstantSequence( v, 0 );
   v_host = -1;
   inplaceExclusiveScan( v_view, 0, size, std::plus<>{}, (ValueType) 0 );
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host = -1;
   inplaceExclusiveScan( v_view, 0, size, std::plus<>{}, (ValueType) 0 );
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], i ) << "i = " << i;

   setLinearSequence( v );
   v_host = -1;
   inplaceExclusiveScan( v_view, 0, size, std::plus<>{}, (ValueType) 0 );
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

   ////
   // With CUDA, perform tests with multiple CUDA grids.
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      Algorithms::detail::CudaScanKernelLauncher< Algorithms::detail::ScanType::Exclusive, ValueType, IndexType >::maxGridSize() = 3;

      setConstantSequence( v, 0 );
      v_host = -1;
      inplaceExclusiveScan( v, 0, size, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::detail::ScanType::Exclusive, ValueType, IndexType >::gridsCount() ), 1 );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

      setConstantSequence( v, 1 );
      v_host = -1;
      inplaceExclusiveScan( v, 0, size, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::detail::ScanType::Exclusive, ValueType, IndexType >::gridsCount() ), 1 );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], i ) << "i = " << i;

      setLinearSequence( v );
      v_host = -1;
      inplaceExclusiveScan( v, 0, size, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::detail::ScanType::Exclusive, ValueType, IndexType >::gridsCount() ), 1 );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

      // test views
      setConstantSequence( v, 0 );
      v_host = -1;
      inplaceExclusiveScan( v_view, 0, size, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::detail::ScanType::Exclusive, ValueType, IndexType >::gridsCount() ), 1 );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

      setConstantSequence( v, 1 );
      v_host = -1;
      inplaceExclusiveScan( v_view, 0, size, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::detail::ScanType::Exclusive, ValueType, IndexType >::gridsCount() ), 1 );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], i ) << "i = " << i;

      setLinearSequence( v );
      v_host = -1;
      inplaceExclusiveScan( v_view, 0, size, std::plus<>{}, (ValueType) 0 );
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::detail::ScanType::Exclusive, ValueType, IndexType >::gridsCount() ), 1 );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

      Algorithms::detail::CudaScanKernelLauncher< Algorithms::detail::ScanType::Exclusive, ValueType, IndexType >::resetMaxGridSize();
#endif
   }
}

// TODO: test scan with custom begin and end parameters

#endif // HAVE_GTEST

#include "../main.h"
