/***************************************************************************
                          tnlDenseMatrixTester.h  -  description
                             -------------------
    begin                : Nov 30, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLDENSEMATRIXTESTER_H_
#define TNLDENSEMATRIXTESTER_H_

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <TNL/matrices/tnlDenseMatrix.h>
#include <TNL/File.h>
#include <TNL/Vectors/Vector.h>

#ifdef HAVE_CUDA
template< typename RealType, typename IndexType >
__global__ void setElementFastTestKernel( tnlDenseMatrix< RealType, tnlCuda, IndexType >* matrix );
template< typename RealType, typename IndexType >
__global__ void addElementFastTestKernel( tnlDenseMatrix< RealType, tnlCuda, IndexType >* matrix );
template< typename RealType, typename IndexType >
__global__ void setRowFastTestKernel( tnlDenseMatrix< RealType, tnlCuda, IndexType >* matrix,
                                      const IndexType* columns,
                                      const RealType* values,
                                      const IndexType numberOfElements );
#endif

using namespace TNL;

template< typename RealType, typename Device, typename IndexType >
class tnlDenseMatrixTester : public CppUnit :: TestCase
{
   public:
   typedef tnlDenseMatrix< RealType, Device, IndexType > MatrixType;
   typedef Vectors::Vector< RealType, Device, IndexType > VectorType;
   typedef tnlDenseMatrixTester< RealType, Device, IndexType > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;

   tnlDenseMatrixTester(){};

   virtual
   ~tnlDenseMatrixTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlDenseMatrixTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "setDimensionsTest", &TesterType::setDimensionsTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setElementTest", &TesterType::setElementTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setElementFastTest", &TesterType::setElementFastTest ) );
      suiteOfTests -> addTest( new TestCallerType( "addElementTest", &TesterType::addElementTest ) );
      suiteOfTests -> addTest( new TestCallerType( "addElementFastTest", &TesterType::addElementTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setRowTest", &TesterType::setRowTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setRowFastTest", &TesterType::setRowFastTest ) );
      suiteOfTests -> addTest( new TestCallerType( "vectorProductTest", &TesterType::vectorProductTest ) );
      suiteOfTests -> addTest( new TestCallerType( "addMatrixTest", &TesterType::addMatrixTest ) );
      suiteOfTests -> addTest( new TestCallerType( "matrixProductTest", &TesterType::matrixProductTest ) );
      suiteOfTests -> addTest( new TestCallerType( "matrixTranspositionTest", &TesterType::matrixTranspositionTest ) );

      return suiteOfTests;
   }

   void setDimensionsTest()
   {
      MatrixType m;
      m.setDimensions( 10, 20 );
      CPPUNIT_ASSERT( m.getRows() == 10 );
      CPPUNIT_ASSERT( m.getColumns() == 20 );
   }

   void setElementTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      for( int i = 0; i < 10; i++ )
         m.setElement( i, i, i );
      for( int i = 0; i < 10; i++ )
         CPPUNIT_ASSERT( m.getElement( i, i ) == i );
   }

   void setElementFastTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      if( Device::getDevice() == tnlHostDevice )
      {
         for( int i = 0; i < 10; i++ )
            m.setElementFast( i, i,  i );
      }
      if( Device::getDevice() == tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_m = tnlCuda::passToDevice( m );
         CPPUNIT_ASSERT( checkCudaDevice );
         setElementFastTestKernel<<< 1, 16 >>>( kernel_m );
         CPPUNIT_ASSERT( checkCudaDevice );
         tnlCuda::freeFromDevice( kernel_m );
         CPPUNIT_ASSERT( checkCudaDevice );
#endif
      }
      for( int i = 0; i < 10; i++ )
         CPPUNIT_ASSERT( m.getElement( i, i ) == i );
   }


   void addElementTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      for( int i = 0; i < 10; i++ )
         m.setElement( i, i, i );
      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            m.addElement( i, j, 1 );
      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, i ) == i + 1 );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 1 );
   }

   void addElementFastTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      for( int i = 0; i < 10; i++ )
         m.setElement( i, i, i );
      if( Device::getDevice() == tnlHostDevice )
      {
         for( int i = 0; i < 10; i++ )
            for( int j = 0; j < 10; j++ )
               m.addElementFast( i, j, 1 );
      }
      if( Device::getDevice() == tnlCudaDevice )
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_m = tnlCuda::passToDevice( m );
         CPPUNIT_ASSERT( checkCudaDevice );
         addElementFastTestKernel<<< 1, 128 >>>( kernel_m );
         CPPUNIT_ASSERT( checkCudaDevice );
         tnlCuda::freeFromDevice( kernel_m );
         CPPUNIT_ASSERT( checkCudaDevice );
#endif
      }
      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, i ) == i + 1 );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 1 );
   }


   void setRowTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      m.setValue( 0.0 );
      for( int i = 0; i < 10; i++ )
         m.setElement( i, i, i );

      Vectors::Vector< IndexType, tnlHost, IndexType > columns;
      Vectors::Vector< RealType, tnlHost, IndexType > values;
      columns.setSize( 10 );
      values.setSize( 10 );
      for( IndexType i = 0; i < 10; i++ )
      {
         columns[ i ] = i;
         values[ i ] = i;
      }
      m.setRow( 5, columns.getData(), values.getData(), 10 );

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
         {
            if( i == 5 )
               CPPUNIT_ASSERT( m.getElement( i, j ) == j );
            else
               if( i == j )
                  CPPUNIT_ASSERT( m.getElement( i, i ) == i );
               else
                  CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
         }
   }

   void setRowFastTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      m.setValue( 0.0 );
      for( int i = 0; i < 10; i++ )
         m.setElement( i, i, i );

      Vectors::Vector< IndexType, Device, IndexType > columns;
      Vectors::Vector< RealType, Device, IndexType > values;
      columns.setSize( 10 );
      values.setSize( 10 );
      for( IndexType i = 0; i < 10; i++ )
      {
         columns.setElement( i,  i );
         values.setElement( i, i );
      }
      if( Device::getDevice() == tnlHostDevice)
         m.setRowFast( 5, columns.getData(), values.getData(), 10 );
      if( Device::getDevice() == tnlCudaDevice)
      {
#ifdef HAVE_CUDA
         MatrixType* kernel_m = tnlCuda::passToDevice( m );
         CPPUNIT_ASSERT( checkCudaDevice );
         setRowFastTestKernel<<< 1, 128 >>>( kernel_m, columns.getData(), values.getData(), ( IndexType ) 10 );
         CPPUNIT_ASSERT( checkCudaDevice );
         tnlCuda::freeFromDevice( kernel_m );
         CPPUNIT_ASSERT( checkCudaDevice );
#endif
      }

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
         {
            if( i == 5 )
               CPPUNIT_ASSERT( m.getElement( i, j ) == j );
            else
               if( i == j )
                  CPPUNIT_ASSERT( m.getElement( i, i ) == i );
               else
                  CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
         }
   }


   void vectorProductTest()
   {
      const int size = 10;
      VectorType v, w;
      v.setSize( size );
      w.setSize( size );
      MatrixType m;
      m.setDimensions( size, size );
      m.setValue( 0.0 );
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
      const int size = 10;
      MatrixType m;
      m.setDimensions( 10, 10 );
      for( int i = 0; i < size; i++ )
         for( int j = 0; j < size; j++ )
            m.setElement( i, j, i*size + j );

      MatrixType m2;
      m2.setLike( m );
      m2.setValue( 3.0 );
      m2.addMatrix( m );

      for( int i = 0; i < size; i++ )
         for( int j = 0; j < size; j++ )
            CPPUNIT_ASSERT( m2.getElement( i, j ) == m.getElement( i, j ) + 3.0 );

      m2.addMatrix( m, 0.5, 0.0 );

      for( int i = 0; i < size; i++ )
         for( int j = 0; j < size; j++ )
            CPPUNIT_ASSERT( m2.getElement( i, j ) == 0.5*m.getElement( i, j ) );
   }

   void matrixProductTest()
   {
      const int size = 10;
      MatrixType m1, m2, m3;
      m1.setDimensions( 10, 10 );
      m2.setLike( m1 );
      m3.setLike( m1 );
      for( int i = 0; i < size; i++ )
         for( int j = 0; j < size; j++ )
         {
            m1.setElement( i, j, i*size + j );
            m2.setElement( i, j, ( i == j ) );
         }
      m3. template getMatrixProduct< MatrixType, MatrixType, 32 >( m1, m2 );

      for( int i = 0; i < size; i++ )
         for( int j = 0; j < size; j++ )
            CPPUNIT_ASSERT( m3.getElement( i, j ) == m1.getElement( i, j) );

   }

   void matrixTranspositionTest()
   {
      const int alignedSize = 64;
      MatrixType m;
      m.setDimensions( alignedSize, alignedSize );
      for( int i = 0; i < alignedSize; i++ )
         for( int j = 0; j < alignedSize; j++ )
            m.setElement( i, j, i*alignedSize + j );

      MatrixType mTransposed;
      mTransposed.setLike( m );
      mTransposed. template getTransposition< MatrixType, 32 >( m );

      for( int i = 0; i < alignedSize; i++ )
         for( int j = 0; j < alignedSize; j++ )
            CPPUNIT_ASSERT( m.getElement( i, j ) == mTransposed.getElement( j, i ) );

      const int nonAlignedSize = 50;
      m.setDimensions( nonAlignedSize, nonAlignedSize );
      for( int i = 0; i < nonAlignedSize; i++ )
         for( int j = 0; j < nonAlignedSize; j++ )
            m.setElement( i, j, i*nonAlignedSize + j );

      mTransposed.setLike( m );
      mTransposed. template getTransposition< MatrixType, 32 >( m );

      for( int i = 0; i < nonAlignedSize; i++ )
         for( int j = 0; j < nonAlignedSize; j++ )
            CPPUNIT_ASSERT( m.getElement( i, j ) == mTransposed.getElement( j, i ) );
   }
};

#ifdef HAVE_CUDA
template< typename RealType, typename IndexType >
__global__ void setElementFastTestKernel( tnlDenseMatrix< RealType, tnlCuda, IndexType >* matrix )
{
   if( threadIdx.x < matrix->getRows() )
      matrix->setElementFast( threadIdx.x, threadIdx.x, threadIdx.x );
}
template< typename RealType, typename IndexType >
__global__ void addElementFastTestKernel( tnlDenseMatrix< RealType, tnlCuda, IndexType >* matrix )
{

   const IndexType column = threadIdx.x;
   if( threadIdx.x < matrix->getRows() )
      matrix->addElementFast( threadIdx.x, threadIdx.x, 1 );
}

template< typename RealType, typename IndexType >
__global__ void setRowFastTestKernel( tnlDenseMatrix< RealType, tnlCuda, IndexType >* matrix,
                                      const IndexType* columns,
                                      const RealType* values,
                                      const IndexType numberOfElements )
{
   if( threadIdx.x == 0 )
      matrix->setRowFast( 5, columns, values, numberOfElements );
}



#endif /* HAVE_CUDA */

#endif /* HAVE_CPPUNIT */

#endif /* TNLDENSEMATRIXTESTER_H_ */
