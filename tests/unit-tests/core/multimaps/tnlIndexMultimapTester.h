/***************************************************************************
                          tnlIndexMultimapTester.h  -  description
                             -------------------
    begin                : Sep 10, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLINDEXMULTIMAPTESTER_H_
#define TNLINDEXMULTIMAPTESTER_H_

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <TNL/File.h>
#include <TNL/Vectors/Vector.h>

using namespace TNL;

template< typename Multimap,
          typename TestSetup >
class tnlIndexMultimapTesterSetter
{
   public:

   static bool setup( Multimap& multimap )
   {
      return true;
   }
};


#ifdef HAVE_CUDA
template< typename IndexMultimapType >
__global__ void tnlIndexMultimapTester__setElementFastTestCudaKernel( IndexMultimapType* graph,
                                                                     bool* testResult );
template< typename IndexMultimapType >
__global__ void tnlIndexMultimapTester__setElementFast_DiagonalIndexMultimapTestCudaKernel( IndexMultimapType* graph,
                                                                                    bool* testResult );

template< typename IndexMultimapType >
__global__ void tnlIndexMultimapTester__setElementFast_DenseIndexMultimapTestCudaKernel1( IndexMultimapType* graph,
                                                                                  bool* testResult );

template< typename IndexMultimapType >
__global__ void tnlIndexMultimapTester__setElementFast_DenseIndexMultimapTestCudaKernel2( IndexMultimapType* graph,
                                                                                  bool* testResult );

template< typename IndexMultimapType >
__global__ void tnlIndexMultimapTester__setElementFast_LowerTriangularIndexMultimapTestCudaKernel1( IndexMultimapType* graph,
                                                                                            bool* testResult );

template< typename IndexMultimapType >
__global__ void tnlIndexMultimapTester__setElementFast_LowerTriangularIndexMultimapTestCudaKernel2( IndexMultimapType* graph,
                                                                                            bool* testResult );

template< typename IndexMultimapType >
__global__ void tnlIndexMultimapTester__setRowFast_DiagonalIndexMultimapTestCudaKernel( IndexMultimapType* graph,
                                                                                bool* testResult );

template< typename IndexMultimapType >
__global__ void tnlIndexMultimapTester__setRowFast_DenseIndexMultimapTestCudaKernel1( IndexMultimapType* graph,
                                                                              bool* testResult );

template< typename IndexMultimapType >
__global__ void tnlIndexMultimapTester__setRowFast_DenseIndexMultimapTestCudaKernel2( IndexMultimapType* graph,
                                                                              bool* testResult );

template< typename IndexMultimapType >
__global__ void tnlIndexMultimapTester__setRowFast_LowerTriangularIndexMultimapTestCudaKernel( IndexMultimapType* graph,
                                                                                       bool* testResult );

#endif

class tnlIndexMultimapTestDefaultSetup
{};

template< typename IndexMultimap,
          typename IndexMultimapSetup = tnlIndexMultimapTestDefaultSetup >
class tnlIndexMultimapTester : public CppUnit :: TestCase
{
   public:
      typedef IndexMultimap                                                    IndexMultimapType;
      typedef typename IndexMultimap::DeviceType                               DeviceType;
      typedef typename IndexMultimap::IndexType                                IndexType;
      typedef tnlIndexMultimapTester< IndexMultimapType, IndexMultimapSetup >              TesterType;
      typedef tnlIndexMultimapTesterSetter< IndexMultimapType, IndexMultimapSetup > IndexMultimapSetter;
      typedef typename CppUnit::TestCaller< TesterType >                 TestCallerType;
 
      typedef typename IndexMultimapType::ValuesAllocationVectorType  ValuesAllocationVectorType;
      typedef typename IndexMultimapType::ValuesAccessorType                  ValuesAccessorType;

      tnlIndexMultimapTester(){};

      virtual
      ~tnlIndexMultimapTester(){};

      static CppUnit :: Test* suite()
      {
         String testSuiteName( "tnlIndexMultimapTester< " );
         testSuiteName += IndexMultimapType::getType() + " >";

         CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( testSuiteName.getString() );
         CppUnit :: TestResult result;

         suiteOfTests->addTest( new TestCallerType( "setRangesTest", &TesterType::setRangesTest ) );
         //suiteOfTests->addTest( new TestCallerType( "setLikeTest", &TesterType::setLikeTest ) );
         suiteOfTests->addTest( new TestCallerType( "setElementTest", &TesterType::setElementTest ) );
         /*suiteOfTests->addTest( new TestCallerType( "setElementFastTest", &TesterType::setElementFastTest ) );
         suiteOfTests->addTest( new TestCallerType( "setElement_DiagonalIndexMultimapTest", &TesterType::setElement_DiagonalIndexMultimapTest ) );
         suiteOfTests->addTest( new TestCallerType( "setElementFast_DiagonalIndexMultimapTest", &TesterType::setElementFast_DiagonalIndexMultimapTest ) );
         suiteOfTests->addTest( new TestCallerType( "setElement_DenseIndexMultimapTest", &TesterType::setElement_DenseIndexMultimapTest ) );
         suiteOfTests->addTest( new TestCallerType( "setElementFast_DenseIndexMultimapTest", &TesterType::setElementFast_DenseIndexMultimapTest ) );
         suiteOfTests->addTest( new TestCallerType( "setElement_LowerTriangularIndexMultimapTest", &TesterType::setElement_LowerTriangularIndexMultimapTest ) );
         suiteOfTests->addTest( new TestCallerType( "setElementFast_LowerTriangularIndexMultimapTest", &TesterType::setElementFast_LowerTriangularIndexMultimapTest ) );
         suiteOfTests->addTest( new TestCallerType( "setRow_DiagonalIndexMultimapTest", &TesterType::setRow_DiagonalIndexMultimapTest ) );
         suiteOfTests->addTest( new TestCallerType( "setRowFast_DiagonalIndexMultimapTest", &TesterType::setRowFast_DiagonalIndexMultimapTest ) );
         suiteOfTests->addTest( new TestCallerType( "setRow_DenseIndexMultimapTest", &TesterType::setRow_DenseIndexMultimapTest ) );
         suiteOfTests->addTest( new TestCallerType( "setRowFast_DenseIndexMultimapTest", &TesterType::setRowFast_DenseIndexMultimapTest ) );
         suiteOfTests->addTest( new TestCallerType( "setRow_LowerTriangularIndexMultimapTest", &TesterType::setRow_LowerTriangularIndexMultimapTest ) );
         suiteOfTests->addTest( new TestCallerType( "setRowFast_LowerTriangularIndexMultimapTest", &TesterType::setRowFast_LowerTriangularIndexMultimapTest ) );
         suiteOfTests->addTest( new TestCallerType( "addElementTest", &TesterType::addElementTest ) );
         suiteOfTests->addTest( new TestCallerType( "vectorProduct_DiagonalIndexMultimapTest", &TesterType::vectorProduct_DiagonalIndexMultimapTest ) );
         suiteOfTests->addTest( new TestCallerType( "vectorProduct_DenseIndexMultimapTest", &TesterType::vectorProduct_DenseIndexMultimapTest ) );
         suiteOfTests->addTest( new TestCallerType( "vectorProduct_LowerTriangularIndexMultimapTest", &TesterType::vectorProduct_LowerTriangularIndexMultimapTest ) );
         /*suiteOfTests -> addTest( new TestCallerType( "graphTranspositionTest", &TesterType::graphTranspositionTest ) );
         suiteOfTests -> addTest( new TestCallerType( "addIndexMultimapTest", &TesterType::addIndexMultimapTest ) );*/

         return suiteOfTests;
      }

      void setRangesTest()
      {
         IndexMultimapType n;
         IndexMultimapSetter::setup( n );
         n.setRanges( 10, 10 );
         CPPUNIT_ASSERT( n.getKeysRange() == 10 );
         CPPUNIT_ASSERT( n.getValuesRange() == 10 );
      }

      /*void setLikeTest()
      {
         IndexMultimapType m1, m2;
         IndexMultimapSetter::setup( m1 );
         IndexMultimapSetter::setup( m2 );
         m1.setDimensions( 10, 10 );
         IndexVector rowLengths;
         rowLengths.setSize( m1.getRows() );
         rowLengths.setValue( 5 );
         m1.setCompressedRowsLengths( rowLengths );
         m2.setLike( m1 );
         CPPUNIT_ASSERT( m1.getRows() == m2.getRows() );
      }*/

      /****
       * Set element tests
       */
      void setElementTest()
      {
         IndexMultimapType n;
         IndexMultimapSetter::setup( n );
         n.setRanges( 10, 10 );
 
         ValuesAllocationVectorType portsAllocationVector;
         portsAllocationVector.setSize( n.getKeysRange() );
         portsAllocationVector.setValue( 7 );
         n.allocate( portsAllocationVector );

         ValuesAccessorType p = n.getValues( 0 );
         for( int i = 0; i < 7; i++ )
         {
            p.setOutput( i, i );
            //CPPUNIT_ASSERT( n.setPort( 0, i, i ) );
         }

         //CPPUNIT_ASSERT( m.setElement( 0, 8, 8 ) == false );

         for( int i = 0; i < 7; i++ )
            CPPUNIT_ASSERT( p.getOutput( i ) == i );
      }
#ifdef UNDEF

   void setElementFastTest()
   {
      IndexMultimapType m;
      IndexMultimapSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setCompressedRowsLengths( rowLengths );

      if( DeviceType::getDevice() == tnlHostDevice )
      {
         for( int i = 0; i < 7; i++ )
            CPPUNIT_ASSERT( m.setElementFast( 0, i, i ) );
         //CPPUNIT_ASSERT( m.setElementFast( 0, 8, 8 ) == false );
      }

      if( DeviceType::getDevice() == tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         IndexMultimapType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlIndexMultimapTester__setElementFastTestCudaKernel< IndexMultimapType >
                                                            <<< cudaGridSize, cudaBlockSize >>>
                                                            ( kernel_graph,
                                                              kernel_testResult );
         CPPUNIT_ASSERT( tnlCuda::passFromDevice( kernel_testResult ) );
         tnlCuda::freeFromDevice( kernel_graph );
         tnlCuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 0; i < 7; i++ )
         CPPUNIT_ASSERT( m.getElement( 0, i ) == i );
   }

   void setElement_DiagonalIndexMultimapTest()
   {
      IndexMultimapType m;
      IndexMultimapSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setCompressedRowsLengths( rowLengths );

      for( int i = 0; i < 10; i++ )
         m.setElement( i, i, i );

      for( int i = 0; i < 10; i++ )
      {
         for( int j = 0; j < 10; j++ )
         {
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
         }
      }
   }

   void setElementFast_DiagonalIndexMultimapTest()
   {
      IndexMultimapType m;
      IndexMultimapSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setCompressedRowsLengths( rowLengths );

      if( DeviceType::DeviceType == ( int ) tnlHostDevice )
      {
         for( int i = 0; i < 10; i++ )
            m.setElementFast( i, i, i );
      }
      if( DeviceType::DeviceType == ( int ) tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         IndexMultimapType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlIndexMultimapTester__setElementFast_DiagonalIndexMultimapTestCudaKernel< IndexMultimapType >
                                                                           <<< cudaGridSize, cudaBlockSize >>>
                                                                           ( kernel_graph,
                                                                             kernel_testResult );
         CPPUNIT_ASSERT( tnlCuda::passFromDevice( kernel_testResult ) );
         tnlCuda::freeFromDevice( kernel_graph );
         tnlCuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 0; i < 10; i++ )
      {
         for( int j = 0; j < 10; j++ )
         {
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
         }
      }
   }

   void setElement_DenseIndexMultimapTest()
   {
      IndexMultimapType m;
      IndexMultimapSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 10 );
      m.setCompressedRowsLengths( rowLengths );

      for( int i = 0; i < 10; i++ )
         m.setElement( i, i, i );
      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            m.addElement( i, j, 1, 0.5 );

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, j ) == 1.0+0.5*i );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 1.0 );

      m.reset();
      m.setDimensions( 10, 10 );
      m.setCompressedRowsLengths( rowLengths );
      for( int i = 9; i >= 0; i-- )
         for( int j = 9; j >= 0; j-- )
            m.setElement( i, j, i+j );

      for( int i = 9; i >= 0; i-- )
         for( int j = 9; j >= 0; j-- )
            CPPUNIT_ASSERT( m.getElement( i, j ) == i+j );
   }

   void setElementFast_DenseIndexMultimapTest()
   {
      IndexMultimapType m;
      IndexMultimapSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 10 );
      m.setCompressedRowsLengths( rowLengths );

      if( DeviceType::DeviceType == ( int ) tnlHostDevice )
      {
         for( int i = 0; i < 10; i++ )
            m.setElementFast( i, i, i );
         for( int i = 0; i < 10; i++ )
            for( int j = 0; j < 10; j++ )
               m.addElementFast( i, j, 1, 0.5 );
      }
      if( DeviceType::DeviceType == ( int ) tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         IndexMultimapType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlIndexMultimapTester__setElementFast_DenseIndexMultimapTestCudaKernel1< IndexMultimapType >
                                                                         <<< cudaGridSize, cudaBlockSize >>>
                                                                         ( kernel_graph,
                                                                           kernel_testResult );
         CPPUNIT_ASSERT( tnlCuda::passFromDevice( kernel_testResult ) );
         tnlCuda::freeFromDevice( kernel_graph );
         tnlCuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, j ) == 1.0+0.5*i );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 1.0 );

      m.reset();
      m.setDimensions( 10, 10 );
      m.setCompressedRowsLengths( rowLengths );
      if( DeviceType::DeviceType == ( int ) tnlHostDevice )
      {
         for( int i = 9; i >= 0; i-- )
            for( int j = 9; j >= 0; j-- )
               m.setElementFast( i, j, i+j );
      }
      if( DeviceType::DeviceType == ( int ) tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         IndexMultimapType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlIndexMultimapTester__setElementFast_DenseIndexMultimapTestCudaKernel2< IndexMultimapType >
                                                                         <<< cudaGridSize, cudaBlockSize >>>
                                                                         ( kernel_graph,
                                                                           kernel_testResult );
         CPPUNIT_ASSERT( tnlCuda::passFromDevice( kernel_testResult ) );
         tnlCuda::freeFromDevice( kernel_graph );
         tnlCuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 9; i >= 0; i-- )
         for( int j = 9; j >= 0; j-- )
            CPPUNIT_ASSERT( m.getElement( i, j ) == i+j );
   }


   void setElement_LowerTriangularIndexMultimapTest()
   {
      IndexMultimapType m;
      IndexMultimapSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      for( int i = 0; i < 10; i++ )
         rowLengths.setElement( i, i+1 );
      m.setCompressedRowsLengths( rowLengths );

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j <= i; j++ )
            m.setElement( i, j, i + j );

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( j <= i )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i + j );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );

      m.reset();
      m.setDimensions( 10, 10 );
      m.setCompressedRowsLengths( rowLengths );
      for( int i = 9; i >= 0; i-- )
         for( int j = i; j >= 0; j-- )
            m.setElement( i, j, i + j );

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( j <= i )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i + j );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
   }

   void setElementFast_LowerTriangularIndexMultimapTest()
   {
      IndexMultimapType m;
      IndexMultimapSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      for( int i = 0; i < 10; i++ )
         rowLengths.setElement( i, i+1 );
      m.setCompressedRowsLengths( rowLengths );

      if( DeviceType::DeviceType == ( int ) tnlHostDevice )
      {
         for( int i = 0; i < 10; i++ )
            for( int j = 0; j <= i; j++ )
               m.setElementFast( i, j, i + j );
      }
      if( DeviceType::DeviceType == ( int ) tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         IndexMultimapType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlIndexMultimapTester__setElementFast_LowerTriangularIndexMultimapTestCudaKernel1< IndexMultimapType >
                                                                                   <<< cudaGridSize, cudaBlockSize >>>
                                                                                   ( kernel_graph,
                                                                                     kernel_testResult );
         CPPUNIT_ASSERT( tnlCuda::passFromDevice( kernel_testResult ) );
         tnlCuda::freeFromDevice( kernel_graph );
         tnlCuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( j <= i )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i + j );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );

      m.reset();
      m.setDimensions( 10, 10 );
      m.setCompressedRowsLengths( rowLengths );
      if( DeviceType::DeviceType == ( int ) tnlHostDevice )
      {
         for( int i = 9; i >= 0; i-- )
            for( int j = i; j >= 0; j-- )
               m.setElementFast( i, j, i + j );
      }
      if( DeviceType::DeviceType == ( int ) tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         IndexMultimapType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlIndexMultimapTester__setElementFast_LowerTriangularIndexMultimapTestCudaKernel2< IndexMultimapType >
                                                                                   <<< cudaGridSize, cudaBlockSize >>>
                                                                                   ( kernel_graph,
                                                                                     kernel_testResult );
         CPPUNIT_ASSERT( tnlCuda::passFromDevice( kernel_testResult ) );
         tnlCuda::freeFromDevice( kernel_graph );
         tnlCuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( j <= i )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i + j );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
   }


   void addElementTest()
   {
      IndexMultimapType m;
      IndexMultimapSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setCompressedRowsLengths( rowLengths );
      for( int i = 0; i < 10; i++ )
         m.setElement( i, i, i );
      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( abs( i - j ) <= 1 )
               m.addElement( i, j, 1 );

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, i ) == i + 1 );
            else
               if( abs( i - j ) == 1 )
                  CPPUNIT_ASSERT( m.getElement( i, j ) == 1 );
               else
                  CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
   }

   /****
    * Set row tests
    */
   void setRow_DiagonalIndexMultimapTest()
   {
      IndexMultimapType m;
      IndexMultimapSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setCompressedRowsLengths( rowLengths );
      RealType values[ 1 ];
      IndexType columnIndexes[ 1 ];

      for( int i = 0; i < 10; i++ )
      {
         values[ 0 ] = i;
         columnIndexes[ 0 ] = i;
         m.setRow( i, columnIndexes, values, 1 );
      }

      for( int i = 0; i < 10; i++ )
      {
         for( int j = 0; j < 10; j++ )
         {
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
         }
      }
   }

   void setRowFast_DiagonalIndexMultimapTest()
   {
      IndexMultimapType m;
      IndexMultimapSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setCompressedRowsLengths( rowLengths );


      if( DeviceType::DeviceType == ( int ) tnlHostDevice )
      {
         RealType values[ 1 ];
         IndexType columnIndexes[ 1 ];
         for( int i = 0; i < 10; i++ )
         {
            values[ 0 ] = i;
            columnIndexes[ 0 ] = i;
            m.setRowFast( i, columnIndexes, values, 1 );
         }
      }
      if( DeviceType::DeviceType == ( int ) tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         IndexMultimapType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         tnlIndexMultimapTester__setRowFast_DiagonalIndexMultimapTestCudaKernel< IndexMultimapType >
                                                                       <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                                                       ( kernel_graph,
                                                                         kernel_testResult );
         CPPUNIT_ASSERT( tnlCuda::passFromDevice( kernel_testResult ) );
         tnlCuda::freeFromDevice( kernel_graph );
         tnlCuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 0; i < 10; i++ )
      {
         for( int j = 0; j < 10; j++ )
         {
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
         }
      }
   }

   void setRow_DenseIndexMultimapTest()
   {
      IndexMultimapType m;
      IndexMultimapSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 10 );
      m.setCompressedRowsLengths( rowLengths );
      RealType values[ 10 ];
      IndexType columnIndexes[ 10 ];

      for( int i = 0; i < 10; i++ )
         columnIndexes[ i ] = i;
      for( int i = 0; i < 10; i++ )
      {
         for( int j = 0; j < 10; j++ )
            if( i == j )
               values[ i ] = 1.0 + 0.5 * j;
            else
               values[ j ] = 1.0;

         m.setRow( i, columnIndexes, values, 10 );
      }

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, j ) == 1.0+0.5*i );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 1.0 );

      m.reset();
      m.setDimensions( 10, 10 );
      m.setCompressedRowsLengths( rowLengths );
      for( int i = 9; i >= 0; i-- )
      {
         for( int j = 9; j >= 0; j-- )
            values[ j ] = i+j;
         m.setRow( i, columnIndexes, values, 10 );
      }

      for( int i = 9; i >= 0; i-- )
         for( int j = 9; j >= 0; j-- )
            CPPUNIT_ASSERT( m.getElement( i, j ) == i+j );
   }

   void setRowFast_DenseIndexMultimapTest()
   {
      IndexMultimapType m;
      IndexMultimapSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 10 );
      m.setCompressedRowsLengths( rowLengths );

      RealType values[ 10 ];
      IndexType columnIndexes[ 10 ];
      for( int i = 0; i < 10; i++ )
         columnIndexes[ i ] = i;

      if( DeviceType::DeviceType == ( int ) tnlHostDevice )
      {
         for( int i = 0; i < 10; i++ )
         {
            for( int j = 0; j < 10; j++ )
               if( i == j )
                  values[ i ] = 1.0 + 0.5 * j;
               else
                  values[ j ] = 1.0;

            m.setRowFast( i, columnIndexes, values, 10 );
         }
      }
      if( DeviceType::DeviceType == ( int ) tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         IndexMultimapType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         tnlIndexMultimapTester__setRowFast_DenseIndexMultimapTestCudaKernel1< IndexMultimapType >
                                                                        <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                                                        ( kernel_graph,
                                                                          kernel_testResult );
         CPPUNIT_ASSERT( tnlCuda::passFromDevice( kernel_testResult ) );
         tnlCuda::freeFromDevice( kernel_graph );
         tnlCuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, j ) == 1.0+0.5*i );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 1.0 );

      m.reset();
      m.setDimensions( 10, 10 );
      m.setCompressedRowsLengths( rowLengths );

      if( DeviceType::DeviceType == ( int ) tnlHostDevice )
      {
         for( int i = 9; i >= 0; i-- )
         {
            for( int j = 9; j >= 0; j-- )
               values[ j ] = i+j;
            m.setRowFast( i, columnIndexes, values, 10 );
         }
      }
      if( DeviceType::DeviceType == ( int ) tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         IndexMultimapType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         tnlIndexMultimapTester__setRowFast_DenseIndexMultimapTestCudaKernel2< IndexMultimapType >
                                                                     <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                                                     ( kernel_graph,
                                                                       kernel_testResult );
         CPPUNIT_ASSERT( tnlCuda::passFromDevice( kernel_testResult ) );
         tnlCuda::freeFromDevice( kernel_graph );
         tnlCuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 9; i >= 0; i-- )
         for( int j = 9; j >= 0; j-- )
            CPPUNIT_ASSERT( m.getElement( i, j ) == i+j );
   }

   void setRow_LowerTriangularIndexMultimapTest()
   {
      IndexMultimapType m;
      IndexMultimapSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      for( int i = 0; i < 10; i++ )
         rowLengths.setElement( i, i+1 );
      m.setCompressedRowsLengths( rowLengths );


      RealType values[ 10 ];
      IndexType columnIndexes[ 10 ];

      for( int i = 0; i < 10; i++ )
         columnIndexes[ i ] = i;

      for( int i = 0; i < 10; i++ )
      {
         for( int j = 0; j <= i; j++ )
            values[ j ] = i + j;
         m.setRow( i, columnIndexes, values, i + 1 );
      }

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( j <= i )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i + j );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );

      m.reset();
      m.setDimensions( 10, 10 );
      m.setCompressedRowsLengths( rowLengths );
      for( int i = 9; i >= 0; i-- )
      {
         for( int j = i; j >= 0; j-- )
            values[ j ] = i + j;
         m.setRow( i, columnIndexes, values, i + 1 );
      }

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( j <= i )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i + j );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
   }

   void setRowFast_LowerTriangularIndexMultimapTest()
   {
      IndexMultimapType m;
      IndexMultimapSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      for( int i = 0; i < 10; i++ )
         rowLengths.setElement( i, i+1 );
      m.setCompressedRowsLengths( rowLengths );


      RealType values[ 10 ];
      IndexType columnIndexes[ 10 ];
      for( int i = 0; i < 10; i++ )
         columnIndexes[ i ] = i;

      if( DeviceType::DeviceType == ( int ) tnlHostDevice )
      {
         for( int i = 0; i < 10; i++ )
         {
            for( int j = 0; j <= i; j++ )
               values[ j ] = i + j;
            m.setRowFast( i, columnIndexes, values, i + 1 );
         }
      }
      if( DeviceType::DeviceType == ( int ) tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         IndexMultimapType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         tnlIndexMultimapTester__setRowFast_LowerTriangularIndexMultimapTestCudaKernel< IndexMultimapType >
                                                                              <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                                                              ( kernel_graph,
                                                                                kernel_testResult );
         CPPUNIT_ASSERT( tnlCuda::passFromDevice( kernel_testResult ) );
         tnlCuda::freeFromDevice( kernel_graph );
         tnlCuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( j <= i )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i + j );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );

      m.reset();
      m.setDimensions( 10, 10 );
      m.setCompressedRowsLengths( rowLengths );

      if( DeviceType::DeviceType == ( int ) tnlHostDevice )
      {
         for( int i = 9; i >= 0; i-- )
         {
            for( int j = i; j >= 0; j-- )
               values[ j ] = i + j;
            m.setRowFast( i, columnIndexes, values, i + 1 );
         }
      }
      if( DeviceType::DeviceType == ( int ) tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         IndexMultimapType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         tnlIndexMultimapTester__setRowFast_LowerTriangularIndexMultimapTestCudaKernel< IndexMultimapType >
                                                                              <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                                                              ( kernel_graph,
                                                                                kernel_testResult );
         CPPUNIT_ASSERT( tnlCuda::passFromDevice( kernel_testResult ) );
         tnlCuda::freeFromDevice( kernel_graph );
         tnlCuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( j <= i )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i + j );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
   }


   void vectorProduct_DiagonalIndexMultimapTest()
   {
      const int size = 10;
      VectorType v, w;
      v.setSize( size );
      w.setSize( size );
      IndexMultimapType m;
      IndexMultimapSetter::setup( m );
      m.setDimensions( size, size );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setCompressedRowsLengths( rowLengths );
      for( int i = 0; i < size; i++ )
      {
         v.setElement( i, i );
         m.setElement( i, i, i );
      }
      m.vectorProduct( v, w );

      for( int i = 0; i < size; i++ )
         CPPUNIT_ASSERT( w.getElement( i ) == i*i );
   }

   void vectorProduct_DenseIndexMultimapTest()
   {
      const int size = 10;
      VectorType v, w;
      v.setSize( size );
      w.setSize( size );
      IndexMultimapType m;
      IndexMultimapSetter::setup( m );
      m.setDimensions( size, size );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( size );
      m.setCompressedRowsLengths( rowLengths );
      for( int i = 0; i < size; i++ )
      {
         for( int j = 0; j < size; j++ )
            m.setElement( i, j, i );
         v.setElement( i, 1 );
      }
      m.vectorProduct( v, w );

      for( int i = 0; i < size; i++ )
         CPPUNIT_ASSERT( w.getElement( i ) == i*size );
   }

   void vectorProduct_LowerTriangularIndexMultimapTest()
   {
      const int size = 10;
      VectorType v, w;
      v.setSize( size );
      w.setSize( size );
      IndexMultimapType m;
      IndexMultimapSetter::setup( m );
      m.setDimensions( size, size );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( size );
      m.setCompressedRowsLengths( rowLengths );
      for( int i = 0; i < size; i++ )
      {
         for( int j = 0; j <= i; j++ )
            m.setElement( i, j, i );
         v.setElement( i, 1 );
      }
      m.vectorProduct( v, w );

      for( int i = 0; i < size; i++ )
         CPPUNIT_ASSERT( w.getElement( i ) == i*( i + 1 ) );
   }


   void addIndexMultimapTest()
   {
   }

   void graphTranspositionTest()
   {
   }
#endif

};

#ifdef HAVE_CUDA
   template< typename IndexMultimapType >
   __global__ void tnlIndexMultimapTester__setElementFastTestCudaKernel( IndexMultimapType* graph,
                                                                        bool* testResult )
   {
      if( threadIdx.x == 0 )
      {
         for( int i = 0; i < 7; i++ )
            if( graph->setElementFast( 0, i, i ) != true )
               testResult = false;
         if( graph->setElementFast( 0, 8, 8 ) == true )
            testResult = false;
      }
   }

   template< typename IndexMultimapType >
   __global__ void tnlIndexMultimapTester__setElementFast_DiagonalIndexMultimapTestCudaKernel( IndexMultimapType* graph,
                                                                                       bool* testResult )
   {
      if( threadIdx.x < graph->getRows() )
         graph->setElementFast( threadIdx.x, threadIdx.x, threadIdx.x );
   }

   template< typename IndexMultimapType >
   __global__ void tnlIndexMultimapTester__setElementFast_DenseIndexMultimapTestCudaKernel1( IndexMultimapType* graph,
                                                                                     bool* testResult )
   {
      const typename IndexMultimapType::IndexType i = threadIdx.x;
      if( i < graph->getRows() )
      {
         graph->setElementFast( i, i, i );
         for( int j = 0; j < graph->getColumns(); j++ )
            graph->addElementFast( i, j, 1, 0.5 );
      }
   }

   template< typename IndexMultimapType >
   __global__ void tnlIndexMultimapTester__setElementFast_DenseIndexMultimapTestCudaKernel2( IndexMultimapType* graph,
                                                                                     bool* testResult )
   {
      const typename IndexMultimapType::IndexType i = threadIdx.x;
      if( i < graph->getRows() )
      {
         for( int j = graph->getColumns() -1; j >= 0; j-- )
            graph->setElementFast( i, j, i + j );
      }
   }

   template< typename IndexMultimapType >
   __global__ void tnlIndexMultimapTester__setElementFast_LowerTriangularIndexMultimapTestCudaKernel1( IndexMultimapType* graph,
                                                                                               bool* testResult )
   {
      const typename IndexMultimapType::IndexType i = threadIdx.x;
      if( i < graph->getRows() )
      {
         for( int j = 0; j <= i; j++ )
            graph->setElementFast( i, j, i + j );
      }
   }

   template< typename IndexMultimapType >
   __global__ void tnlIndexMultimapTester__setElementFast_LowerTriangularIndexMultimapTestCudaKernel2( IndexMultimapType* graph,
                                                                                               bool* testResult )
   {
      const typename IndexMultimapType::IndexType i = threadIdx.x;
      if( i < graph->getRows() )
      {
         for( int j = i; j >= 0; j-- )
            graph->setElementFast( i, j, i + j );
      }
   }

   /****
    * Set row tests kernels
    */
   template< typename IndexMultimapType >
   __global__ void tnlIndexMultimapTester__setRowFast_DiagonalIndexMultimapTestCudaKernel( IndexMultimapType* graph,
                                                                                   bool* testResult )
   {
      typedef typename IndexMultimapType::RealType RealType;
      typedef typename IndexMultimapType::IndexType IndexType;

      const IndexType row = threadIdx.x;
      if( row >= graph->getRows() )
         return;

      IndexType* columnIndexes = getSharedMemory< IndexType >();
      RealType* valuesBase = ( RealType* ) & columnIndexes[ graph->getColumns() ];
      RealType* values = &valuesBase[ row ];

      columnIndexes[ row ] = row;
      values[ 0 ] = row;

      graph->setRowFast( row, &columnIndexes[ row ], values, 1 );
   }

   template< typename IndexMultimapType >
   __global__ void tnlIndexMultimapTester__setRowFast_DenseIndexMultimapTestCudaKernel1( IndexMultimapType* graph,
                                                                                 bool* testResult )
   {
      typedef typename IndexMultimapType::RealType RealType;
      typedef typename IndexMultimapType::IndexType IndexType;

      const IndexType row = threadIdx.x;
      if( row >= graph->getRows() )
         return;

      IndexType* columnIndexes = getSharedMemory< IndexType >();
      RealType* valuesBase = ( RealType* ) & columnIndexes[ graph->getColumns() ];
      RealType* values  = &valuesBase[ row * graph->getColumns() ];

      columnIndexes[ row ] = row;

      for( int i = 0; i < graph->getColumns(); i++ )
      {
         if( i == row )
            values[ i ] = 1.0 + 0.5 * i;
         else
            values[ i ] = 1.0;
      }
      graph->setRowFast( row, columnIndexes, values, 10 );
   }

   template< typename IndexMultimapType >
   __global__ void tnlIndexMultimapTester__setRowFast_DenseIndexMultimapTestCudaKernel2( IndexMultimapType* graph,
                                                                                 bool* testResult )
   {
      typedef typename IndexMultimapType::RealType RealType;
      typedef typename IndexMultimapType::IndexType IndexType;

      const IndexType row = threadIdx.x;
      if( row >= graph->getRows() )
         return;

      IndexType* columnIndexes = getSharedMemory< IndexType >();
      RealType* valuesBase = ( RealType* ) & columnIndexes[ graph->getColumns() ];
      RealType* values  = &valuesBase[ row * graph->getColumns() ];

      columnIndexes[ row ] = row;

      for( int i = 0; i < graph->getColumns(); i++ )
      {
            values[ i ] = row + i;
      }
      graph->setRowFast( row, columnIndexes, values, 10 );
   }

   template< typename IndexMultimapType >
   __global__ void tnlIndexMultimapTester__setRowFast_LowerTriangularIndexMultimapTestCudaKernel( IndexMultimapType* graph,
                                                                                          bool* testResult )
   {
      typedef typename IndexMultimapType::RealType RealType;
      typedef typename IndexMultimapType::IndexType IndexType;

      const IndexType row = threadIdx.x;
      if( row >= graph->getRows() )
         return;

      IndexType* columnIndexes = getSharedMemory< IndexType >();
      RealType* valuesBase = ( RealType* ) & columnIndexes[ graph->getColumns() ];
      RealType* values  = &valuesBase[ row * graph->getColumns() ];

      columnIndexes[ row ] = row;

      for( int i = 0; i <= row; i++ )
      {
            values[ i ] = row + i;
      }
      graph->setRowFast( row, columnIndexes, values, row + 1 );
   }

#endif


#endif

#endif /* TNLINDEXMULTIMAPTESTER_H_ */
