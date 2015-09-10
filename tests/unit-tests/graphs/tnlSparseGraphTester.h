/***************************************************************************
                          tnlSparseGraphTester.h  -  description
                             -------------------
    begin                : Sep 10, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLSPARSEGRAPHTESTER_H_
#define TNLSPARSEGRAPHTESTER_H_

template< typename Graph,
          typename TestSetup >
class tnlSparseGraphTesterGraphSetter
{
   public:

   static bool setup( Graph& graph )
   {
      return true;
   }
};

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/tnlFile.h>
#include <core/vectors/tnlVector.h>

#ifdef HAVE_CUDA
template< typename GraphType >
__global__ void tnlSparseGraphTester__setElementFastTestCudaKernel( GraphType* graph,
                                                                     bool* testResult );
template< typename GraphType >
__global__ void tnlSparseGraphTester__setElementFast_DiagonalGraphTestCudaKernel( GraphType* graph,
                                                                                    bool* testResult );

template< typename GraphType >
__global__ void tnlSparseGraphTester__setElementFast_DenseGraphTestCudaKernel1( GraphType* graph,
                                                                                  bool* testResult );

template< typename GraphType >
__global__ void tnlSparseGraphTester__setElementFast_DenseGraphTestCudaKernel2( GraphType* graph,
                                                                                  bool* testResult );

template< typename GraphType >
__global__ void tnlSparseGraphTester__setElementFast_LowerTriangularGraphTestCudaKernel1( GraphType* graph,
                                                                                            bool* testResult );

template< typename GraphType >
__global__ void tnlSparseGraphTester__setElementFast_LowerTriangularGraphTestCudaKernel2( GraphType* graph,
                                                                                            bool* testResult );

template< typename GraphType >
__global__ void tnlSparseGraphTester__setRowFast_DiagonalGraphTestCudaKernel( GraphType* graph,
                                                                                bool* testResult );

template< typename GraphType >
__global__ void tnlSparseGraphTester__setRowFast_DenseGraphTestCudaKernel1( GraphType* graph,
                                                                              bool* testResult );

template< typename GraphType >
__global__ void tnlSparseGraphTester__setRowFast_DenseGraphTestCudaKernel2( GraphType* graph,
                                                                              bool* testResult );

template< typename GraphType >
__global__ void tnlSparseGraphTester__setRowFast_LowerTriangularGraphTestCudaKernel( GraphType* graph,
                                                                                       bool* testResult );

#endif

class tnlSparseGraphTestDefaultSetup
{};

template< typename Graph,
          typename GraphSetup = tnlSparseGraphTestDefaultSetup >
class tnlSparseGraphTester : public CppUnit :: TestCase
{
   public:
   typedef Graph GraphType;
   typedef typename Graph::RealType RealType;
   typedef typename Graph::DeviceType DeviceType;
   typedef typename Graph::IndexType IndexType;
   typedef tnlVector< RealType, DeviceType, IndexType > VectorType;
   typedef tnlVector< IndexType, DeviceType, IndexType > IndexVector;
   typedef tnlSparseGraphTester< GraphType, GraphSetup > TesterType;
   typedef tnlSparseGraphTesterGraphSetter< GraphType, GraphSetup > GraphSetter;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;

   tnlSparseGraphTester(){};

   virtual
   ~tnlSparseGraphTester(){};

   static CppUnit :: Test* suite()
   {
      tnlString testSuiteName( "tnlSparseGraphTester< " );
      testSuiteName += GraphType::getType() + " >";

      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( testSuiteName.getString() );
      CppUnit :: TestResult result;

      suiteOfTests->addTest( new TestCallerType( "setDimensionsTest", &TesterType::setDimensionsTest ) );
      suiteOfTests->addTest( new TestCallerType( "setLikeTest", &TesterType::setLikeTest ) );
      suiteOfTests->addTest( new TestCallerType( "setElementTest", &TesterType::setElementTest ) );
      suiteOfTests->addTest( new TestCallerType( "setElementFastTest", &TesterType::setElementFastTest ) );
      suiteOfTests->addTest( new TestCallerType( "setElement_DiagonalGraphTest", &TesterType::setElement_DiagonalGraphTest ) );
      suiteOfTests->addTest( new TestCallerType( "setElementFast_DiagonalGraphTest", &TesterType::setElementFast_DiagonalGraphTest ) );
      suiteOfTests->addTest( new TestCallerType( "setElement_DenseGraphTest", &TesterType::setElement_DenseGraphTest ) );
      suiteOfTests->addTest( new TestCallerType( "setElementFast_DenseGraphTest", &TesterType::setElementFast_DenseGraphTest ) );
      suiteOfTests->addTest( new TestCallerType( "setElement_LowerTriangularGraphTest", &TesterType::setElement_LowerTriangularGraphTest ) );
      suiteOfTests->addTest( new TestCallerType( "setElementFast_LowerTriangularGraphTest", &TesterType::setElementFast_LowerTriangularGraphTest ) );
      suiteOfTests->addTest( new TestCallerType( "setRow_DiagonalGraphTest", &TesterType::setRow_DiagonalGraphTest ) );
      suiteOfTests->addTest( new TestCallerType( "setRowFast_DiagonalGraphTest", &TesterType::setRowFast_DiagonalGraphTest ) );
      suiteOfTests->addTest( new TestCallerType( "setRow_DenseGraphTest", &TesterType::setRow_DenseGraphTest ) );
      suiteOfTests->addTest( new TestCallerType( "setRowFast_DenseGraphTest", &TesterType::setRowFast_DenseGraphTest ) );
      suiteOfTests->addTest( new TestCallerType( "setRow_LowerTriangularGraphTest", &TesterType::setRow_LowerTriangularGraphTest ) );
      suiteOfTests->addTest( new TestCallerType( "setRowFast_LowerTriangularGraphTest", &TesterType::setRowFast_LowerTriangularGraphTest ) );
      suiteOfTests->addTest( new TestCallerType( "addElementTest", &TesterType::addElementTest ) );
      suiteOfTests->addTest( new TestCallerType( "vectorProduct_DiagonalGraphTest", &TesterType::vectorProduct_DiagonalGraphTest ) );
      suiteOfTests->addTest( new TestCallerType( "vectorProduct_DenseGraphTest", &TesterType::vectorProduct_DenseGraphTest ) );
      suiteOfTests->addTest( new TestCallerType( "vectorProduct_LowerTriangularGraphTest", &TesterType::vectorProduct_LowerTriangularGraphTest ) );
      /*suiteOfTests -> addTest( new TestCallerType( "graphTranspositionTest", &TesterType::graphTranspositionTest ) );
      suiteOfTests -> addTest( new TestCallerType( "addGraphTest", &TesterType::addGraphTest ) );*/

      return suiteOfTests;
   }

   void setDimensionsTest()
   {
      GraphType m;
      GraphSetter::setup( m );
      m.setDimensions( 10, 10 );
      CPPUNIT_ASSERT( m.getRows() == 10 );
      CPPUNIT_ASSERT( m.getColumns() == 10 );
   }

   void setLikeTest()
   {
      GraphType m1, m2;
      GraphSetter::setup( m1 );
      GraphSetter::setup( m2 );
      m1.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m1.getRows() );
      rowLengths.setValue( 5 );
      m1.setCompressedRowsLengths( rowLengths );
      m2.setLike( m1 );
      CPPUNIT_ASSERT( m1.getRows() == m2.getRows() );
   }

   /****
    * Set element tests
    */
   void setElementTest()
   {
      GraphType m;
      GraphSetter::setup( m );
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setCompressedRowsLengths( rowLengths );

      for( int i = 0; i < 7; i++ )
         CPPUNIT_ASSERT( m.setElement( 0, i, i ) );

      //CPPUNIT_ASSERT( m.setElement( 0, 8, 8 ) == false );

      for( int i = 0; i < 7; i++ )
         CPPUNIT_ASSERT( m.getElement( 0, i ) == i );
   }


   void setElementFastTest()
   {
      GraphType m;
      GraphSetter::setup( m );
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
         GraphType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlSparseGraphTester__setElementFastTestCudaKernel< GraphType >
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

   void setElement_DiagonalGraphTest()
   {
      GraphType m;
      GraphSetter::setup( m );
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

   void setElementFast_DiagonalGraphTest()
   {
      GraphType m;
      GraphSetter::setup( m );
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
         GraphType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlSparseGraphTester__setElementFast_DiagonalGraphTestCudaKernel< GraphType >
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

   void setElement_DenseGraphTest()
   {
      GraphType m;
      GraphSetter::setup( m );
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

   void setElementFast_DenseGraphTest()
   {
      GraphType m;
      GraphSetter::setup( m );
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
         GraphType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlSparseGraphTester__setElementFast_DenseGraphTestCudaKernel1< GraphType >
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
         GraphType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlSparseGraphTester__setElementFast_DenseGraphTestCudaKernel2< GraphType >
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


   void setElement_LowerTriangularGraphTest()
   {
      GraphType m;
      GraphSetter::setup( m );
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

   void setElementFast_LowerTriangularGraphTest()
   {
      GraphType m;
      GraphSetter::setup( m );
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
         GraphType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlSparseGraphTester__setElementFast_LowerTriangularGraphTestCudaKernel1< GraphType >
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
         GraphType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlSparseGraphTester__setElementFast_LowerTriangularGraphTestCudaKernel2< GraphType >
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
      GraphType m;
      GraphSetter::setup( m );
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
   void setRow_DiagonalGraphTest()
   {
      GraphType m;
      GraphSetter::setup( m );
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

   void setRowFast_DiagonalGraphTest()
   {
      GraphType m;
      GraphSetter::setup( m );
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
         GraphType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         tnlSparseGraphTester__setRowFast_DiagonalGraphTestCudaKernel< GraphType >
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

   void setRow_DenseGraphTest()
   {
      GraphType m;
      GraphSetter::setup( m );
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

   void setRowFast_DenseGraphTest()
   {
      GraphType m;
      GraphSetter::setup( m );
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
         GraphType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         tnlSparseGraphTester__setRowFast_DenseGraphTestCudaKernel1< GraphType >
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
         GraphType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         tnlSparseGraphTester__setRowFast_DenseGraphTestCudaKernel2< GraphType >
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

   void setRow_LowerTriangularGraphTest()
   {
      GraphType m;
      GraphSetter::setup( m );
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

   void setRowFast_LowerTriangularGraphTest()
   {
      GraphType m;
      GraphSetter::setup( m );
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
         GraphType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         tnlSparseGraphTester__setRowFast_LowerTriangularGraphTestCudaKernel< GraphType >
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
         GraphType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         tnlSparseGraphTester__setRowFast_LowerTriangularGraphTestCudaKernel< GraphType >
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


   void vectorProduct_DiagonalGraphTest()
   {
      const int size = 10;
      VectorType v, w;
      v.setSize( size );
      w.setSize( size );
      GraphType m;
      GraphSetter::setup( m );
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

   void vectorProduct_DenseGraphTest()
   {
      const int size = 10;
      VectorType v, w;
      v.setSize( size );
      w.setSize( size );
      GraphType m;
      GraphSetter::setup( m );
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

   void vectorProduct_LowerTriangularGraphTest()
   {
      const int size = 10;
      VectorType v, w;
      v.setSize( size );
      w.setSize( size );
      GraphType m;
      GraphSetter::setup( m );
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


   void addGraphTest()
   {
   }

   void graphTranspositionTest()
   {
   }
};

#ifdef HAVE_CUDA
   template< typename GraphType >
   __global__ void tnlSparseGraphTester__setElementFastTestCudaKernel( GraphType* graph,
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

   template< typename GraphType >
   __global__ void tnlSparseGraphTester__setElementFast_DiagonalGraphTestCudaKernel( GraphType* graph,
                                                                                       bool* testResult )
   {
      if( threadIdx.x < graph->getRows() )
         graph->setElementFast( threadIdx.x, threadIdx.x, threadIdx.x );
   }

   template< typename GraphType >
   __global__ void tnlSparseGraphTester__setElementFast_DenseGraphTestCudaKernel1( GraphType* graph,
                                                                                     bool* testResult )
   {
      const typename GraphType::IndexType i = threadIdx.x;
      if( i < graph->getRows() )
      {
         graph->setElementFast( i, i, i );
         for( int j = 0; j < graph->getColumns(); j++ )
            graph->addElementFast( i, j, 1, 0.5 );
      }
   }

   template< typename GraphType >
   __global__ void tnlSparseGraphTester__setElementFast_DenseGraphTestCudaKernel2( GraphType* graph,
                                                                                     bool* testResult )
   {
      const typename GraphType::IndexType i = threadIdx.x;
      if( i < graph->getRows() )
      {
         for( int j = graph->getColumns() -1; j >= 0; j-- )
            graph->setElementFast( i, j, i + j );
      }
   }

   template< typename GraphType >
   __global__ void tnlSparseGraphTester__setElementFast_LowerTriangularGraphTestCudaKernel1( GraphType* graph,
                                                                                               bool* testResult )
   {
      const typename GraphType::IndexType i = threadIdx.x;
      if( i < graph->getRows() )
      {
         for( int j = 0; j <= i; j++ )
            graph->setElementFast( i, j, i + j );
      }
   }

   template< typename GraphType >
   __global__ void tnlSparseGraphTester__setElementFast_LowerTriangularGraphTestCudaKernel2( GraphType* graph,
                                                                                               bool* testResult )
   {
      const typename GraphType::IndexType i = threadIdx.x;
      if( i < graph->getRows() )
      {
         for( int j = i; j >= 0; j-- )
            graph->setElementFast( i, j, i + j );
      }
   }

   /****
    * Set row tests kernels
    */
   template< typename GraphType >
   __global__ void tnlSparseGraphTester__setRowFast_DiagonalGraphTestCudaKernel( GraphType* graph,
                                                                                   bool* testResult )
   {
      typedef typename GraphType::RealType RealType;
      typedef typename GraphType::IndexType IndexType;

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

   template< typename GraphType >
   __global__ void tnlSparseGraphTester__setRowFast_DenseGraphTestCudaKernel1( GraphType* graph,
                                                                                 bool* testResult )
   {
      typedef typename GraphType::RealType RealType;
      typedef typename GraphType::IndexType IndexType;

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

   template< typename GraphType >
   __global__ void tnlSparseGraphTester__setRowFast_DenseGraphTestCudaKernel2( GraphType* graph,
                                                                                 bool* testResult )
   {
      typedef typename GraphType::RealType RealType;
      typedef typename GraphType::IndexType IndexType;

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

   template< typename GraphType >
   __global__ void tnlSparseGraphTester__setRowFast_LowerTriangularGraphTestCudaKernel( GraphType* graph,
                                                                                          bool* testResult )
   {
      typedef typename GraphType::RealType RealType;
      typedef typename GraphType::IndexType IndexType;

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

#endif /* TNLSPARSEMATRIXTESTER_H_ */
