/***************************************************************************
                          tnlNetworkTester.h  -  description
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

#ifndef TNLSPARSENETWORKTESTER_H_
#define TNLSPARSENETWORKTESTER_H_

template< typename Network,
          typename TestSetup >
class tnlNetworkTesterNetworkSetter
{
   public:

   static bool setup( Network& graph )
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
template< typename NetworkType >
__global__ void tnlNetworkTester__setElementFastTestCudaKernel( NetworkType* graph,
                                                                     bool* testResult );
template< typename NetworkType >
__global__ void tnlNetworkTester__setElementFast_DiagonalNetworkTestCudaKernel( NetworkType* graph,
                                                                                    bool* testResult );

template< typename NetworkType >
__global__ void tnlNetworkTester__setElementFast_DenseNetworkTestCudaKernel1( NetworkType* graph,
                                                                                  bool* testResult );

template< typename NetworkType >
__global__ void tnlNetworkTester__setElementFast_DenseNetworkTestCudaKernel2( NetworkType* graph,
                                                                                  bool* testResult );

template< typename NetworkType >
__global__ void tnlNetworkTester__setElementFast_LowerTriangularNetworkTestCudaKernel1( NetworkType* graph,
                                                                                            bool* testResult );

template< typename NetworkType >
__global__ void tnlNetworkTester__setElementFast_LowerTriangularNetworkTestCudaKernel2( NetworkType* graph,
                                                                                            bool* testResult );

template< typename NetworkType >
__global__ void tnlNetworkTester__setRowFast_DiagonalNetworkTestCudaKernel( NetworkType* graph,
                                                                                bool* testResult );

template< typename NetworkType >
__global__ void tnlNetworkTester__setRowFast_DenseNetworkTestCudaKernel1( NetworkType* graph,
                                                                              bool* testResult );

template< typename NetworkType >
__global__ void tnlNetworkTester__setRowFast_DenseNetworkTestCudaKernel2( NetworkType* graph,
                                                                              bool* testResult );

template< typename NetworkType >
__global__ void tnlNetworkTester__setRowFast_LowerTriangularNetworkTestCudaKernel( NetworkType* graph,
                                                                                       bool* testResult );

#endif

class tnlNetworkTestDefaultSetup
{};

template< typename Network,
          typename NetworkSetup = tnlNetworkTestDefaultSetup >
class tnlNetworkTester : public CppUnit :: TestCase
{
   public:
      typedef Network                                                    NetworkType;
      typedef typename Network::DeviceType                               DeviceType;
      typedef typename Network::IndexType                                IndexType;
      typedef tnlNetworkTester< NetworkType, NetworkSetup >              TesterType;
      typedef tnlNetworkTesterNetworkSetter< NetworkType, NetworkSetup > NetworkSetter;
      typedef typename CppUnit::TestCaller< TesterType >                 TestCallerType;
      
      typedef typename NetworkType::PortsAllocationVectorType  PortsAllocationVectorType;
      typedef typename NetworkType::PortsType                  PortsType; 

      tnlNetworkTester(){};

      virtual
      ~tnlNetworkTester(){};

      static CppUnit :: Test* suite()
      {
         tnlString testSuiteName( "tnlNetworkTester< " );
         testSuiteName += NetworkType::getType() + " >";

         CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( testSuiteName.getString() );
         CppUnit :: TestResult result;

         suiteOfTests->addTest( new TestCallerType( "setDimensionsTest", &TesterType::setDimensionsTest ) );
         //suiteOfTests->addTest( new TestCallerType( "setLikeTest", &TesterType::setLikeTest ) );
         suiteOfTests->addTest( new TestCallerType( "setElementTest", &TesterType::setElementTest ) );
         /*suiteOfTests->addTest( new TestCallerType( "setElementFastTest", &TesterType::setElementFastTest ) );
         suiteOfTests->addTest( new TestCallerType( "setElement_DiagonalNetworkTest", &TesterType::setElement_DiagonalNetworkTest ) );
         suiteOfTests->addTest( new TestCallerType( "setElementFast_DiagonalNetworkTest", &TesterType::setElementFast_DiagonalNetworkTest ) );
         suiteOfTests->addTest( new TestCallerType( "setElement_DenseNetworkTest", &TesterType::setElement_DenseNetworkTest ) );
         suiteOfTests->addTest( new TestCallerType( "setElementFast_DenseNetworkTest", &TesterType::setElementFast_DenseNetworkTest ) );
         suiteOfTests->addTest( new TestCallerType( "setElement_LowerTriangularNetworkTest", &TesterType::setElement_LowerTriangularNetworkTest ) );
         suiteOfTests->addTest( new TestCallerType( "setElementFast_LowerTriangularNetworkTest", &TesterType::setElementFast_LowerTriangularNetworkTest ) );
         suiteOfTests->addTest( new TestCallerType( "setRow_DiagonalNetworkTest", &TesterType::setRow_DiagonalNetworkTest ) );
         suiteOfTests->addTest( new TestCallerType( "setRowFast_DiagonalNetworkTest", &TesterType::setRowFast_DiagonalNetworkTest ) );
         suiteOfTests->addTest( new TestCallerType( "setRow_DenseNetworkTest", &TesterType::setRow_DenseNetworkTest ) );
         suiteOfTests->addTest( new TestCallerType( "setRowFast_DenseNetworkTest", &TesterType::setRowFast_DenseNetworkTest ) );
         suiteOfTests->addTest( new TestCallerType( "setRow_LowerTriangularNetworkTest", &TesterType::setRow_LowerTriangularNetworkTest ) );
         suiteOfTests->addTest( new TestCallerType( "setRowFast_LowerTriangularNetworkTest", &TesterType::setRowFast_LowerTriangularNetworkTest ) );
         suiteOfTests->addTest( new TestCallerType( "addElementTest", &TesterType::addElementTest ) );
         suiteOfTests->addTest( new TestCallerType( "vectorProduct_DiagonalNetworkTest", &TesterType::vectorProduct_DiagonalNetworkTest ) );
         suiteOfTests->addTest( new TestCallerType( "vectorProduct_DenseNetworkTest", &TesterType::vectorProduct_DenseNetworkTest ) );
         suiteOfTests->addTest( new TestCallerType( "vectorProduct_LowerTriangularNetworkTest", &TesterType::vectorProduct_LowerTriangularNetworkTest ) );
         /*suiteOfTests -> addTest( new TestCallerType( "graphTranspositionTest", &TesterType::graphTranspositionTest ) );
         suiteOfTests -> addTest( new TestCallerType( "addNetworkTest", &TesterType::addNetworkTest ) );*/

         return suiteOfTests;
      }

      void setDimensionsTest()
      {
         NetworkType n;
         NetworkSetter::setup( n );
         n.setDimensions( 10, 10 );
         CPPUNIT_ASSERT( n.getInputsCount() == 10 );
         CPPUNIT_ASSERT( n.getOutputsCount() == 10 );
      }

      /*void setLikeTest()
      {
         NetworkType m1, m2;
         NetworkSetter::setup( m1 );
         NetworkSetter::setup( m2 );
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
         NetworkType n;
         NetworkSetter::setup( n );
         n.setDimensions( 10, 10 );
         
         PortsAllocationVectorType portsAllocationVector;
         portsAllocationVector.setSize( n.getInputsCount() );
         portsAllocationVector.setValue( 7 );
         n.allocatePorts( portsAllocationVector );

         PortsType p = n.getPorts( 0 );
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
      NetworkType m;
      NetworkSetter::setup( m );
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
         NetworkType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlNetworkTester__setElementFastTestCudaKernel< NetworkType >
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

   void setElement_DiagonalNetworkTest()
   {
      NetworkType m;
      NetworkSetter::setup( m );
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

   void setElementFast_DiagonalNetworkTest()
   {
      NetworkType m;
      NetworkSetter::setup( m );
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
         NetworkType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlNetworkTester__setElementFast_DiagonalNetworkTestCudaKernel< NetworkType >
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

   void setElement_DenseNetworkTest()
   {
      NetworkType m;
      NetworkSetter::setup( m );
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

   void setElementFast_DenseNetworkTest()
   {
      NetworkType m;
      NetworkSetter::setup( m );
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
         NetworkType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlNetworkTester__setElementFast_DenseNetworkTestCudaKernel1< NetworkType >
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
         NetworkType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlNetworkTester__setElementFast_DenseNetworkTestCudaKernel2< NetworkType >
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


   void setElement_LowerTriangularNetworkTest()
   {
      NetworkType m;
      NetworkSetter::setup( m );
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

   void setElementFast_LowerTriangularNetworkTest()
   {
      NetworkType m;
      NetworkSetter::setup( m );
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
         NetworkType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlNetworkTester__setElementFast_LowerTriangularNetworkTestCudaKernel1< NetworkType >
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
         NetworkType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlNetworkTester__setElementFast_LowerTriangularNetworkTestCudaKernel2< NetworkType >
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
      NetworkType m;
      NetworkSetter::setup( m );
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
   void setRow_DiagonalNetworkTest()
   {
      NetworkType m;
      NetworkSetter::setup( m );
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

   void setRowFast_DiagonalNetworkTest()
   {
      NetworkType m;
      NetworkSetter::setup( m );
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
         NetworkType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         tnlNetworkTester__setRowFast_DiagonalNetworkTestCudaKernel< NetworkType >
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

   void setRow_DenseNetworkTest()
   {
      NetworkType m;
      NetworkSetter::setup( m );
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

   void setRowFast_DenseNetworkTest()
   {
      NetworkType m;
      NetworkSetter::setup( m );
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
         NetworkType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         tnlNetworkTester__setRowFast_DenseNetworkTestCudaKernel1< NetworkType >
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
         NetworkType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         tnlNetworkTester__setRowFast_DenseNetworkTestCudaKernel2< NetworkType >
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

   void setRow_LowerTriangularNetworkTest()
   {
      NetworkType m;
      NetworkSetter::setup( m );
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

   void setRowFast_LowerTriangularNetworkTest()
   {
      NetworkType m;
      NetworkSetter::setup( m );
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
         NetworkType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         tnlNetworkTester__setRowFast_LowerTriangularNetworkTestCudaKernel< NetworkType >
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
         NetworkType* kernel_graph = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         int sharedMemory = 100 * ( sizeof( IndexType ) + sizeof( RealType ) );
         tnlNetworkTester__setRowFast_LowerTriangularNetworkTestCudaKernel< NetworkType >
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


   void vectorProduct_DiagonalNetworkTest()
   {
      const int size = 10;
      VectorType v, w;
      v.setSize( size );
      w.setSize( size );
      NetworkType m;
      NetworkSetter::setup( m );
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

   void vectorProduct_DenseNetworkTest()
   {
      const int size = 10;
      VectorType v, w;
      v.setSize( size );
      w.setSize( size );
      NetworkType m;
      NetworkSetter::setup( m );
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

   void vectorProduct_LowerTriangularNetworkTest()
   {
      const int size = 10;
      VectorType v, w;
      v.setSize( size );
      w.setSize( size );
      NetworkType m;
      NetworkSetter::setup( m );
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


   void addNetworkTest()
   {
   }

   void graphTranspositionTest()
   {
   }
#endif

};

#ifdef HAVE_CUDA
   template< typename NetworkType >
   __global__ void tnlNetworkTester__setElementFastTestCudaKernel( NetworkType* graph,
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

   template< typename NetworkType >
   __global__ void tnlNetworkTester__setElementFast_DiagonalNetworkTestCudaKernel( NetworkType* graph,
                                                                                       bool* testResult )
   {
      if( threadIdx.x < graph->getRows() )
         graph->setElementFast( threadIdx.x, threadIdx.x, threadIdx.x );
   }

   template< typename NetworkType >
   __global__ void tnlNetworkTester__setElementFast_DenseNetworkTestCudaKernel1( NetworkType* graph,
                                                                                     bool* testResult )
   {
      const typename NetworkType::IndexType i = threadIdx.x;
      if( i < graph->getRows() )
      {
         graph->setElementFast( i, i, i );
         for( int j = 0; j < graph->getColumns(); j++ )
            graph->addElementFast( i, j, 1, 0.5 );
      }
   }

   template< typename NetworkType >
   __global__ void tnlNetworkTester__setElementFast_DenseNetworkTestCudaKernel2( NetworkType* graph,
                                                                                     bool* testResult )
   {
      const typename NetworkType::IndexType i = threadIdx.x;
      if( i < graph->getRows() )
      {
         for( int j = graph->getColumns() -1; j >= 0; j-- )
            graph->setElementFast( i, j, i + j );
      }
   }

   template< typename NetworkType >
   __global__ void tnlNetworkTester__setElementFast_LowerTriangularNetworkTestCudaKernel1( NetworkType* graph,
                                                                                               bool* testResult )
   {
      const typename NetworkType::IndexType i = threadIdx.x;
      if( i < graph->getRows() )
      {
         for( int j = 0; j <= i; j++ )
            graph->setElementFast( i, j, i + j );
      }
   }

   template< typename NetworkType >
   __global__ void tnlNetworkTester__setElementFast_LowerTriangularNetworkTestCudaKernel2( NetworkType* graph,
                                                                                               bool* testResult )
   {
      const typename NetworkType::IndexType i = threadIdx.x;
      if( i < graph->getRows() )
      {
         for( int j = i; j >= 0; j-- )
            graph->setElementFast( i, j, i + j );
      }
   }

   /****
    * Set row tests kernels
    */
   template< typename NetworkType >
   __global__ void tnlNetworkTester__setRowFast_DiagonalNetworkTestCudaKernel( NetworkType* graph,
                                                                                   bool* testResult )
   {
      typedef typename NetworkType::RealType RealType;
      typedef typename NetworkType::IndexType IndexType;

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

   template< typename NetworkType >
   __global__ void tnlNetworkTester__setRowFast_DenseNetworkTestCudaKernel1( NetworkType* graph,
                                                                                 bool* testResult )
   {
      typedef typename NetworkType::RealType RealType;
      typedef typename NetworkType::IndexType IndexType;

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

   template< typename NetworkType >
   __global__ void tnlNetworkTester__setRowFast_DenseNetworkTestCudaKernel2( NetworkType* graph,
                                                                                 bool* testResult )
   {
      typedef typename NetworkType::RealType RealType;
      typedef typename NetworkType::IndexType IndexType;

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

   template< typename NetworkType >
   __global__ void tnlNetworkTester__setRowFast_LowerTriangularNetworkTestCudaKernel( NetworkType* graph,
                                                                                          bool* testResult )
   {
      typedef typename NetworkType::RealType RealType;
      typedef typename NetworkType::IndexType IndexType;

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
