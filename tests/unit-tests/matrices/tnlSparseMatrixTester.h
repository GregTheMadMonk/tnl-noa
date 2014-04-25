/***************************************************************************
                          tnlSparseMatrixTester.h  -  description
                             -------------------
    begin                : Jul 11, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#ifndef TNLSPARSEMATRIXTESTER_H_
#define TNLSPARSEMATRIXTESTER_H_

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/tnlFile.h>
#include <core/vectors/tnlVector.h>

#ifdef HAVE_CUDA
template< typename MatrixType >
__global__ void tnlSparseMatrixTester__setElementFastTestCudaKernel( MatrixType* matrix,
                                                                     bool* testResult );
template< typename MatrixType >
__global__ void tnlSparseMatrixTester__setElementFast_DiagonalMatrixTestCudaKernel( MatrixType* matrix,
                                                                                    bool* testResult );

template< typename MatrixType >
__global__ void tnlSparseMatrixTester__setElementFast_DenseMatrixTestCudaKernel1( MatrixType* matrix,
                                                                                  bool* testResult );

template< typename MatrixType >
__global__ void tnlSparseMatrixTester__setElementFast_DenseMatrixTestCudaKernel2( MatrixType* matrix,
                                                                                  bool* testResult );


template< typename MatrixType >
__global__ void tnlSparseMatrixTester__setElementFast_LowerTriangularMatrixTestCudaKernel1( MatrixType* matrix,
                                                                                            bool* testResult );

template< typename MatrixType >
__global__ void tnlSparseMatrixTester__setElementFast_LowerTriangularMatrixTestCudaKernel2( MatrixType* matrix,
                                                                                            bool* testResult );


#endif


template< typename Matrix >
class tnlSparseMatrixTester : public CppUnit :: TestCase
{
   public:
   typedef Matrix MatrixType;
   typedef typename Matrix::RealType RealType;
   typedef typename Matrix::DeviceType DeviceType;
   typedef typename Matrix::IndexType IndexType;
   typedef tnlVector< RealType, DeviceType, IndexType > VectorType;
   typedef tnlVector< IndexType, DeviceType, IndexType > IndexVector;
   typedef tnlSparseMatrixTester< MatrixType > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;

   tnlSparseMatrixTester(){};

   virtual
   ~tnlSparseMatrixTester(){};

   static CppUnit :: Test* suite()
   {
      tnlString testSuiteName( "tnlSparseMatrixTester< " );
      testSuiteName += MatrixType::getType() + " >";

      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( testSuiteName.getString() );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "setDimensionsTest", &TesterType::setDimensionsTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setLikeTest", &TesterType::setLikeTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setElementTest", &TesterType::setElementTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setElementFastTest", &TesterType::setElementFastTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setElement_DiagonalMatrixTest", &TesterType::setElement_DiagonalMatrixTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setElementFast_DiagonalMatrixTest", &TesterType::setElementFast_DiagonalMatrixTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setElement_DenseMatrixTest", &TesterType::setElement_DenseMatrixTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setElementFast_DenseMatrixTest", &TesterType::setElementFast_DenseMatrixTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setElement_LowerTriangularMatrixTest", &TesterType::setElement_LowerTriangularMatrixTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setElementFast_LowerTriangularMatrixTest", &TesterType::setElementFast_LowerTriangularMatrixTest ) );
      suiteOfTests -> addTest( new TestCallerType( "addElementTest", &TesterType::addElementTest ) );
      suiteOfTests -> addTest( new TestCallerType( "vectorProductTest", &TesterType::vectorProductTest ) );
      /*suiteOfTests -> addTest( new TestCallerType( "matrixTranspositionTest", &TesterType::matrixTranspositionTest ) );
      suiteOfTests -> addTest( new TestCallerType( "addMatrixTest", &TesterType::addMatrixTest ) );*/

      return suiteOfTests;
   }

   void setDimensionsTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      CPPUNIT_ASSERT( m.getRows() == 10 );
      CPPUNIT_ASSERT( m.getColumns() == 10 );
   }

   void setLikeTest()
   {
      MatrixType m1, m2;
      m1.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m1.getRows() );
      rowLengths.setValue( 5 );
      m1.setRowLengths( rowLengths );
      m2.setLike( m1 );
      CPPUNIT_ASSERT( m1.getRows() == m2.getRows() );
   }

   void setElementTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setRowLengths( rowLengths );

      for( int i = 0; i < 7; i++ )
         CPPUNIT_ASSERT( m.setElement( 0, i, i ) );
      CPPUNIT_ASSERT( m.setElement( 0, 8, 8 ) == false );

      for( int i = 0; i < 7; i++ )
         CPPUNIT_ASSERT( m.getElement( 0, i ) == i );
   }


   void setElementFastTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setRowLengths( rowLengths );

      if( DeviceType::getDevice() == tnlHostDevice )
      {
         for( int i = 0; i < 7; i++ )
            CPPUNIT_ASSERT( m.setElementFast( 0, i, i ) );
         CPPUNIT_ASSERT( m.setElementFast( 0, 8, 8 ) == false );
      }

      if( DeviceType::getDevice() == tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_matrix = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlSparseMatrixTester__setElementFastTestCudaKernel< MatrixType >
                                                            <<< cudaGridSize, cudaBlockSize >>>
                                                            ( kernel_matrix,
                                                              kernel_testResult );
         CPPUNIT_ASSERT( tnlCuda::passFromDevice( kernel_testResult ) );
         tnlCuda::freeFromDevice( kernel_matrix );
         tnlCuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }
      for( int i = 0; i < 7; i++ )
         CPPUNIT_ASSERT( m.getElement( 0, i ) == i );
   }

   void setElement_DiagonalMatrixTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setRowLengths( rowLengths );

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

   void setElementFast_DiagonalMatrixTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setRowLengths( rowLengths );

      if( DeviceType::DeviceType == tnlHostDevice )
      {
         for( int i = 0; i < 10; i++ )
            m.setElementFast( i, i, i );
      }
      if( DeviceType::DeviceType == tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_matrix = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlSparseMatrixTester__setElementFast_DiagonalMatrixTestCudaKernel< MatrixType >
                                                                           <<< cudaGridSize, cudaBlockSize >>>
                                                                           ( kernel_matrix,
                                                                             kernel_testResult );
         CPPUNIT_ASSERT( tnlCuda::passFromDevice( kernel_testResult ) );
         tnlCuda::freeFromDevice( kernel_matrix );
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

   void setElement_DenseMatrixTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 10 );
      m.setRowLengths( rowLengths );

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
      m.setRowLengths( rowLengths );
      for( int i = 9; i >= 0; i-- )
         for( int j = 9; j >= 0; j-- )
            m.setElement( i, j, i+j );

      for( int i = 9; i >= 0; i-- )
         for( int j = 9; j >= 0; j-- )
            CPPUNIT_ASSERT( m.getElement( i, j ) == i+j );
   }

   void setElementFast_DenseMatrixTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 10 );
      m.setRowLengths( rowLengths );

      if( DeviceType::DeviceType == tnlHostDevice )
      {
         for( int i = 0; i < 10; i++ )
            m.setElementFast( i, i, i );
         for( int i = 0; i < 10; i++ )
            for( int j = 0; j < 10; j++ )
               m.addElementFast( i, j, 1, 0.5 );
      }
      if( DeviceType::DeviceType == tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_matrix = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlSparseMatrixTester__setElementFast_DenseMatrixTestCudaKernel1< MatrixType >
                                                                         <<< cudaGridSize, cudaBlockSize >>>
                                                                         ( kernel_matrix,
                                                                           kernel_testResult );
         CPPUNIT_ASSERT( tnlCuda::passFromDevice( kernel_testResult ) );
         tnlCuda::freeFromDevice( kernel_matrix );
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
      m.setRowLengths( rowLengths );
      if( DeviceType::DeviceType == tnlHostDevice )
      {
         for( int i = 9; i >= 0; i-- )
            for( int j = 9; j >= 0; j-- )
               m.setElementFast( i, j, i+j );
      }
      if( DeviceType::DeviceType == tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_matrix = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlSparseMatrixTester__setElementFast_DenseMatrixTestCudaKernel2< MatrixType >
                                                                         <<< cudaGridSize, cudaBlockSize >>>
                                                                         ( kernel_matrix,
                                                                           kernel_testResult );
         CPPUNIT_ASSERT( tnlCuda::passFromDevice( kernel_testResult ) );
         tnlCuda::freeFromDevice( kernel_matrix );
         tnlCuda::freeFromDevice( kernel_testResult );
         checkCudaDevice;
#endif
      }

      for( int i = 9; i >= 0; i-- )
         for( int j = 9; j >= 0; j-- )
            CPPUNIT_ASSERT( m.getElement( i, j ) == i+j );
   }


   void setElement_LowerTriangularMatrixTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      for( int i = 0; i < 10; i++ )
         rowLengths.setElement( i, i+1 );
      m.setRowLengths( rowLengths );

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
      m.setRowLengths( rowLengths );
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

   void setElementFast_LowerTriangularMatrixTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      for( int i = 0; i < 10; i++ )
         rowLengths.setElement( i, i+1 );
      m.setRowLengths( rowLengths );

      if( DeviceType::DeviceType == tnlHostDevice )
      {
         for( int i = 0; i < 10; i++ )
            for( int j = 0; j <= i; j++ )
               m.setElementFast( i, j, i + j );
      }
      if( DeviceType::DeviceType == tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_matrix = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlSparseMatrixTester__setElementFast_LowerTriangularMatrixTestCudaKernel1< MatrixType >
                                                                                   <<< cudaGridSize, cudaBlockSize >>>
                                                                                   ( kernel_matrix,
                                                                                     kernel_testResult );
         CPPUNIT_ASSERT( tnlCuda::passFromDevice( kernel_testResult ) );
         tnlCuda::freeFromDevice( kernel_matrix );
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
      m.setRowLengths( rowLengths );
      if( DeviceType::DeviceType == tnlHostDevice )
      {
         for( int i = 9; i >= 0; i-- )
            for( int j = i; j >= 0; j-- )
               m.setElementFast( i, j, i + j );
      }
      if( DeviceType::DeviceType == tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_matrix = tnlCuda::passToDevice( m );
         bool testResult( true );
         bool* kernel_testResult = tnlCuda::passToDevice( testResult );
         checkCudaDevice;
         dim3 cudaBlockSize( 256 ), cudaGridSize( 1 );
         tnlSparseMatrixTester__setElementFast_LowerTriangularMatrixTestCudaKernel2< MatrixType >
                                                                                   <<< cudaGridSize, cudaBlockSize >>>
                                                                                   ( kernel_matrix,
                                                                                     kernel_testResult );
         CPPUNIT_ASSERT( tnlCuda::passFromDevice( kernel_testResult ) );
         tnlCuda::freeFromDevice( kernel_matrix );
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
      MatrixType m;
      m.setDimensions( 10, 10 );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setRowLengths( rowLengths );
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

   void vectorProductTest()
   {
      const int size = 10;
      VectorType v, w;
      v.setSize( size );
      w.setSize( size );
      MatrixType m;
      m.setDimensions( size, size );
      IndexVector rowLengths;
      rowLengths.setSize( m.getRows() );
      rowLengths.setValue( 7 );
      m.setRowLengths( rowLengths );
      for( int i = 0; i < size; i++ )
      {
         v.setElement( i, i );
         m.setElement( i, i, i );
      }
      m.vectorProduct( v, w );

      for( int i = 0; i < size; i++ )
         CPPUNIT_ASSERT( w.getElement( i ) == i*i );
   }

   void addMatrixTest()
   {
   }

   void matrixTranspositionTest()
   {
   }
};

#ifdef HAVE_CUDA
   template< typename MatrixType >
   __global__ void tnlSparseMatrixTester__setElementFastTestCudaKernel( MatrixType* matrix,
                                                                        bool* testResult )
   {
      if( threadIdx.x == 0 )
      {
         for( int i = 0; i < 7; i++ )
            if( matrix->setElementFast( 0, i, i ) != true )
               testResult = false;
         if( matrix->setElementFast( 0, 8, 8 ) == true )
            testResult = false;
      }
   }

   template< typename MatrixType >
   __global__ void tnlSparseMatrixTester__setElementFast_DiagonalMatrixTestCudaKernel( MatrixType* matrix,
                                                                                       bool* testResult )
   {
      if( threadIdx.x < matrix->getRows() )
         matrix->setElementFast( threadIdx.x, threadIdx.x, threadIdx.x );
   }

   template< typename MatrixType >
   __global__ void tnlSparseMatrixTester__setElementFast_DenseMatrixTestCudaKernel1( MatrixType* matrix,
                                                                                     bool* testResult )
   {
      const typename MatrixType::IndexType i = threadIdx.x;
      if( i < matrix->getRows() )
      {
         matrix->setElementFast( i, i, i );
         for( int j = 0; j < matrix->getColumns(); j++ )
            matrix->addElementFast( i, j, 1, 0.5 );
      }
   }

   template< typename MatrixType >
   __global__ void tnlSparseMatrixTester__setElementFast_DenseMatrixTestCudaKernel2( MatrixType* matrix,
                                                                                     bool* testResult )
   {
      const typename MatrixType::IndexType i = threadIdx.x;
      if( i < matrix->getRows() )
      {
         for( int j = matrix->getColumns() -1; j >= 0; j-- )
            matrix->setElementFast( i, j, i + j );
      }
   }

   template< typename MatrixType >
   __global__ void tnlSparseMatrixTester__setElementFast_LowerTriangularMatrixTestCudaKernel1( MatrixType* matrix,
                                                                                               bool* testResult )
   {
      const typename MatrixType::IndexType i = threadIdx.x;
      if( i < matrix->getRows() )
      {
         for( int j = 0; j <= i; j++ )
            matrix->setElementFast( i, j, i + j );
      }
   }

   template< typename MatrixType >
   __global__ void tnlSparseMatrixTester__setElementFast_LowerTriangularMatrixTestCudaKernel2( MatrixType* matrix,
                                                                                               bool* testResult )
   {
      const typename MatrixType::IndexType i = threadIdx.x;
      if( i < matrix->getRows() )
      {
         for( int j = i; j >= 0; j-- )
            matrix->setElementFast( i, j, i + j );
      }
   }


#endif


#endif

#endif /* TNLSPARSEMATRIXTESTER_H_ */
