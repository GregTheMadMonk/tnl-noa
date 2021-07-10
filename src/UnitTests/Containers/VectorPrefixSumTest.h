/***************************************************************************
                          VectorTest-2.h  -  description
                             -------------------
    begin                : Oct 25, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#ifdef HAVE_GTEST
#include "VectorTestSetup.h"

// should be small enough to have fast tests, but larger than minGPUReductionDataSize
// and large enough to require multiple CUDA blocks for reduction
constexpr int VECTOR_TEST_SIZE = 10000;

TYPED_TEST( VectorTest, scan )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   using DeviceType = typename VectorType::DeviceType;
   using IndexType = typename VectorType::IndexType;
   using HostVectorType = typename VectorType::template Self< RealType, Devices::Sequential >;
   const int size = VECTOR_TEST_SIZE;

   // FIXME: tests should work in all cases
   if( std::is_same< RealType, float >::value )
      return;

   VectorType v( size );
   ViewType v_view( v );
   HostVectorType v_host( size );

   setConstantSequence( v, 0 );
   v_host = -1;
   v.scan();
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host = -1;
   v.scan();
   v_host = v_view;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], i + 1 ) << "i = " << i;

   setLinearSequence( v );
   v_host = -1;
   v.scan();
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

   // test views
   setConstantSequence( v, 0 );
   v_host = -1;
   v_view.scan();
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host = -1;
   v_view.scan();
   v_host = v_view;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], i + 1 ) << "i = " << i;

   setLinearSequence( v );
   v_host = -1;
   v_view.scan();
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

   ////
   // With CUDA, perform tests with multiple CUDA grids.
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, RealType, IndexType >::maxGridSize() = 3;

      setConstantSequence( v, 0 );
      v_host = -1;
      v.scan();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

      setConstantSequence( v, 1 );
      v_host = -1;
      v.scan();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v_view;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], i + 1 ) << "i = " << i;

      setLinearSequence( v );
      v_host = -1;
      v.scan();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

      // test views
      setConstantSequence( v, 0 );
      v_host = -1;
      v_view.scan();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

      setConstantSequence( v, 1 );
      v_host = -1;
      v_view.scan();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v_view;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], i + 1 ) << "i = " << i;

      setLinearSequence( v );
      v_host = -1;
      v_view.scan();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, RealType, IndexType >::gridsCount() ), 1  );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], (i * (i + 1)) / 2 ) << "i = " << i;

      Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Inclusive, RealType, IndexType >::resetMaxGridSize();
#endif
   }
}

TYPED_TEST( VectorTest, exclusiveScan )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   using DeviceType = typename VectorType::DeviceType;
   using IndexType = typename VectorType::IndexType;
   using HostVectorType = typename VectorType::template Self< RealType, Devices::Sequential >;
   const int size = VECTOR_TEST_SIZE;

   // FIXME: tests should work in all cases
   if( std::is_same< RealType, float >::value )
      return;

   VectorType v;
   v.setSize( size );
   ViewType v_view( v );
   HostVectorType v_host( size );

   setConstantSequence( v, 0 );
   v_host = -1;
   v.template scan< Algorithms::ScanType::Exclusive >();
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host = -1;
   v.template scan< Algorithms::ScanType::Exclusive >();
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], i ) << "i = " << i;

   setLinearSequence( v );
   v_host = -1;
   v.template scan< Algorithms::ScanType::Exclusive >();
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

   // test views
   setConstantSequence( v, 0 );
   v_host = -1;
   v_view.template scan< Algorithms::ScanType::Exclusive >();
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

   setConstantSequence( v, 1 );
   v_host = -1;
   v_view.template scan< Algorithms::ScanType::Exclusive >();
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], i ) << "i = " << i;

   setLinearSequence( v );
   v_host = -1;
   v_view.template scan< Algorithms::ScanType::Exclusive >();
   v_host = v;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

   ////
   // With CUDA, perform tests with multiple CUDA grids.
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, RealType, IndexType >::maxGridSize() = 3;

      setConstantSequence( v, 0 );
      v_host = -1;
      v.template scan< Algorithms::ScanType::Exclusive >();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, RealType, IndexType >::gridsCount() ), 1 );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

      setConstantSequence( v, 1 );
      v_host = -1;
      v.template scan< Algorithms::ScanType::Exclusive >();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, RealType, IndexType >::gridsCount() ), 1 );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], i ) << "i = " << i;

      setLinearSequence( v );
      v_host = -1;
      v.template scan< Algorithms::ScanType::Exclusive >();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, RealType, IndexType >::gridsCount() ), 1 );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

      // test views
      setConstantSequence( v, 0 );
      v_host = -1;
      v_view.template scan< Algorithms::ScanType::Exclusive >();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, RealType, IndexType >::gridsCount() ), 1 );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], 0 ) << "i = " << i;

      setConstantSequence( v, 1 );
      v_host = -1;
      v_view.template scan< Algorithms::ScanType::Exclusive >();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, RealType, IndexType >::gridsCount() ), 1 );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], i ) << "i = " << i;

      setLinearSequence( v );
      v_host = -1;
      v_view.template scan< Algorithms::ScanType::Exclusive >();
      EXPECT_GT( ( Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, RealType, IndexType >::gridsCount() ), 1 );
      v_host = v;
      for( int i = 0; i < size; i++ )
         EXPECT_EQ( v_host[ i ], (i * (i - 1)) / 2 ) << "i = " << i;

      Algorithms::detail::CudaScanKernelLauncher< Algorithms::ScanType::Exclusive, RealType, IndexType >::resetMaxGridSize();
#endif
   }
}

// TODO: test scan with custom begin and end parameters

#endif // HAVE_GTEST

#include "../main.h"
