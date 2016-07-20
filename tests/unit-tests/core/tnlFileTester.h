/***************************************************************************
                          tnlFileTester.h  -  description
                             -------------------
    begin                : Oct 24, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <TNL/File.h>
#ifdef HAVE_CUDA
#include <cuda.h>
#endif

using namespace TNL;

class tnlFileTester : public CppUnit :: TestCase
{
   public:
   tnlFileTester(){};

   virtual
   ~tnlFileTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlFileTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlFileTester >(
                               "testWriteAndRead",
                               & tnlFileTester :: testWriteAndRead ) );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlFileTester >(
                               "testWriteAndReadCUDA",
                               & tnlFileTester :: testWriteAndReadCUDA ) );

      return suiteOfTests;
   }

   void testWriteAndRead()
   {
      File file;
      if( ! file. open( String( "test-file.tnl" ), tnlWriteMode ) )
      {
         std::cerr << "Unable to create file test-file.tnl for the testing." << std::endl;
         return;
      }
      int intData( 5 );
#ifdef HAVE_NOT_CXX11
      file. write< int, tnlHost >( &intData );
#else
      file. write( &intData );
#endif
      double doubleData[ 3 ] = { 1.0, 2.0, 3.0 };
#ifdef HAVE_NOT_CXX11
      file. write< double, tnlHost >( doubleData, 3 );
#else
      file. write( doubleData, 3 );
#endif
      if( ! file. close() )
      {
         std::cerr << "Unable to close the file test-file.tnl" << std::endl;
         return;
      }

      if( ! file. open( String( "test-file.tnl" ), tnlReadMode ) )
      {
         std::cerr << "Unable to open the file test-file.tnl for the testing." << std::endl;
         return;
      }
      int newIntData;
      double newDoubleData[ 3 ];
#ifdef HAVE_NOT_CXX11
      file. read< int, tnlHost >( &newIntData );
      file. read< double, tnlHost >( newDoubleData, 3 );
#else
      file. read( &newIntData, 1 );
      file. read( newDoubleData, 3 );
#endif

      CPPUNIT_ASSERT( newIntData == intData );
      for( int i = 0; i < 3; i ++ )
         CPPUNIT_ASSERT( newDoubleData[ i ] == doubleData[ i ] );
   };

   void testWriteAndReadCUDA()
   {
#ifdef HAVE_CUDA
      int intData( 5 );
      float floatData[ 3 ] = { 1.0, 2.0, 3.0 };

      int* cudaIntData;
      float* cudaFloatData;
      cudaMalloc( ( void** ) &cudaIntData, sizeof( int ) );
      cudaMalloc( ( void** ) &cudaFloatData, 3 * sizeof( float ) );
      cudaMemcpy( cudaIntData,
                  &intData,
                  sizeof( int ),
                  cudaMemcpyHostToDevice );
      cudaMemcpy( cudaFloatData,
                  floatData,
                  3 * sizeof( float ),
                  cudaMemcpyHostToDevice );
      File file;
      if( ! file. open( String( "test-file.tnl" ), tnlWriteMode ) )
      {
         std::cerr << "Unable to create file test-file.tnl for the testing." << std::endl;
         return;
      }

      file. write< int, tnlCuda >( cudaIntData );
      file. write< float, tnlCuda, int >( cudaFloatData, 3 );
      if( ! file. close() )
      {
         std::cerr << "Unable to close the file test-file.tnl" << std::endl;
         return;
      }

      if( ! file. open( String( "test-file.tnl" ), tnlReadMode ) )
      {
         std::cerr << "Unable to open the file test-file.tnl for the testing." << std::endl;
         return;
      }
      int newIntData;
      float newFloatData[ 3 ];
      int* newCudaIntData;
      float* newCudaFloatData;
      cudaMalloc( ( void** ) &newCudaIntData, sizeof( int ) );
      cudaMalloc( ( void** ) &newCudaFloatData, 3 * sizeof( float ) );
      file. read< int, tnlCuda >( newCudaIntData, 1 );
      file. read< float, tnlCuda, int >( newCudaFloatData, 3 );
      cudaMemcpy( &newIntData,
                  newCudaIntData,
                  sizeof( int ),
                  cudaMemcpyDeviceToHost );
      cudaMemcpy( newFloatData,
                  newCudaFloatData,
                  3 * sizeof( float ),
                  cudaMemcpyDeviceToHost );

      CPPUNIT_ASSERT( newIntData == intData );
      for( int i = 0; i < 3; i ++ )
         CPPUNIT_ASSERT( newFloatData[ i ] == floatData[ i ] );
#endif
   };


};
