/***************************************************************************
                          tnlEllpackMatrixTester.h  -  description
                             -------------------
    begin                : Jul 31, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLELLPACKMATRIXTESTER_H_
#define TNLELLPACKMATRIXTESTER_H_

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <matrices/tnlEllpackMatrix.h>
#include <core/tnlFile.h>
#include <core/vectors/tnlVector.h>

using namespace TNL;

template< typename RealType, typename Device, typename IndexType >
class tnlEllpackMatrixTester : public CppUnit :: TestCase
{
   public:
   typedef tnlEllpackMatrix< RealType, Device, IndexType > MatrixType;
   typedef tnlVector< RealType, Device, IndexType > VectorType;
   typedef tnlEllpackMatrixTester< RealType, Device, IndexType > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;

   tnlEllpackMatrixTester(){};

   virtual
   ~tnlEllpackMatrixTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlTridiagonalMatrixTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "setDimensionsTest", &TesterType::setDimensionsTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setLikeTest", &TesterType::setLikeTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setElementTest", &TesterType::setElementTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setElement_DiagonalMatrixTest", &TesterType::setElement_DiagonalMatrixTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setElement_DenseMatrixTest", &TesterType::setElement_DenseMatrixTest ) );
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
      m1.setConstantCompressedRowsLengths( 5 );
      m2.setLike( m1 );
      CPPUNIT_ASSERT( m1.getRows() == m2.getRows() );
   }

   void setElementTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      m.setConstantCompressedRowsLengths( 7 );

      for( int i = 0; i < 7; i++ )
         CPPUNIT_ASSERT( m.setElement( 0, i, i ) );
      CPPUNIT_ASSERT( m.setElement( 0, 8, 8 ) == false );
   }

   void setElement_DiagonalMatrixTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      m.setConstantCompressedRowsLengths( 7 );

      for( int i = 0; i < 10; i++ )
         m.setElement( i, i, i );
      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, j ) == i );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
   }

   void setElement_DenseMatrixTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      m.setConstantCompressedRowsLengths( 10 );

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
      m.setConstantCompressedRowsLengths( 10 );
      for( int i = 9; i >= 0; i-- )
         for( int j = 9; j >= 0; j-- )
            m.setElement( i, j, i+j );

      for( int i = 9; i >= 0; i-- )
         for( int j = 9; j >= 0; j-- )
            CPPUNIT_ASSERT( m.getElement( i, j ) == i+j );
   }

   void addElementTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      m.setConstantCompressedRowsLengths( 7 );
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
      m.setConstantCompressedRowsLengths( 7 );
      for( int i = 0; i < size; i++ )
      {
         v.setElement( i, i );
         m.setElement( i, i, i );
      }
      m.vectorProduct( v, w );

      for( int i = 0; i < size; i++ )
         CPPUNIT_ASSERT( w[ i ] == i*i );
   }

   void addMatrixTest()
   {
   }

   void matrixTranspositionTest()
   {
   }
};

#endif



#endif /* TNLELLPACKMATRIXTESTER_H_ */
