/***************************************************************************
                          tnlMultidiagonalMatrixTester.h  -  description
                             -------------------
    begin                : Dec 4, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLMULTIDIAGONALMATRIXTESTER_H_
#define TNLMULTIDIAGONALMATRIXTESTER_H_

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <matrices/tnlMultidiagonalMatrix.h>
#include <core/tnlFile.h>
#include <core/vectors/tnlVector.h>

template< typename RealType, typename Device, typename IndexType >
class tnlMultidiagonalMatrixTester : public CppUnit :: TestCase
{
   public:
   typedef tnlMultidiagonalMatrix< RealType, Device, IndexType > MatrixType;
   typedef tnlVector< RealType, Device, IndexType > VectorType;
   typedef tnlMultidiagonalMatrixTester< RealType, Device, IndexType > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;

   tnlMultidiagonalMatrixTester(){};

   virtual
   ~tnlMultidiagonalMatrixTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlTridiagonalMatrixTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "setDimensionsTest", &TesterType::setDimensionsTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setLikeTest", &TesterType::setLikeTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setElementTest", &TesterType::setElementTest ) );
      suiteOfTests -> addTest( new TestCallerType( "addToElementTest", &TesterType::addToElementTest ) );
      suiteOfTests -> addTest( new TestCallerType( "vectorProductTest", &TesterType::vectorProductTest ) );
      suiteOfTests -> addTest( new TestCallerType( "matrixTranspositionTest", &TesterType::matrixTranspositionTest ) );
      suiteOfTests -> addTest( new TestCallerType( "addMatrixTest", &TesterType::addMatrixTest ) );

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
      m2.setLike( m1 );
      CPPUNIT_ASSERT( m1.getRows() == m2.getRows() );
   }

   void setValueTest()
   {
      const int size( 10 );
      MatrixType m;
      m.setDimensions( size, size );
      m.setValue( 1.0 );
      for( int i = 0; i < size; i++ )
         for( int j = 0; j < size; j++ )
            if( abs( i - j ) <= 1 )
               CPPUNIT_ASSERT( m.getElement( i, j ) == 1.0 );
            else
               CPPUNIT_ASSERT( m.getElement( i, j ) == 0.0 );
   }

   void setElementTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      for( int i = 0; i < 10; i++ )
         m.setElement( i, i, i );
      for( int i = 0; i < 10; i++ )
         CPPUNIT_ASSERT( m.getElement( i, i ) == i );

      for( int i = 0; i < 10; i++ )
         m.setElement( i, i,  i );
      for( int i = 0; i < 10; i++ )
         CPPUNIT_ASSERT( m.getElement( i, i ) == i );
   }

   void addToElementTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      for( int i = 0; i < 10; i++ )
         m.setElement( i, i, i );
      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( abs( i - j ) <= 1 )
               m.addToElement( i, j, 1 );

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
      const int size = 10;
      MatrixType m;
      m.setDimensions( 10, 10 );
      for( int i = 0; i < size; i++ )
         for( int j = 0; j < size; j++ )
            if( abs( i - j ) <= 1 )
               m.setElement( i, j, i*size + j );

      MatrixType m2;
      m2.setLike( m );
      m2.setValue( 3.0 );
      m2.addMatrix( m );

      for( int i = 0; i < size; i++ )
         for( int j = 0; j < size; j++ )
            if( abs( i - j ) <= 1 )
               CPPUNIT_ASSERT( m2.getElement( i, j ) == m.getElement( i, j ) + 3.0 );
            else
               CPPUNIT_ASSERT( m2.getElement( i, j ) == 0.0 );

      m2.addMatrix( m, 0.5, 0.0 );

      for( int i = 0; i < size; i++ )
         for( int j = 0; j < size; j++ )
            if( abs( i - j ) <= 1 )
               CPPUNIT_ASSERT( m2.getElement( i, j ) == 0.5*m.getElement( i, j ) );
            else
               CPPUNIT_ASSERT( m2.getElement( i, j ) == 0.0 );
   }

   void matrixTranspositionTest()
   {
      const int size = 10;
      MatrixType m;
      m.setDimensions( 10, 10 );
      for( int i = 0; i < size; i++ )
         for( int j = 0; j < size; j++ )
            if( abs( i - j ) <= 1 )
               m.setElement( i, j, i*size + j );

      MatrixType mTransposed;
      mTransposed.setLike( m );
      mTransposed.getTransposition( m );

      for( int i = 0; i < size; i++ )
         for( int j = 0; j < size; j++ )
            CPPUNIT_ASSERT( m.getElement( i, j ) == mTransposed.getElement( j, i ) );
   }
};

#endif

#endif /* TNLMULTIDIAGONALMATRIXTESTER_H_ */
