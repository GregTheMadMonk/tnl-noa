/***************************************************************************
                          DenseTester.h  -  description
                             -------------------
    begin                : Dec 2, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TridiagonalTESTER_H_
#define TridiagonalTESTER_H_

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <TNL/Matrices/Tridiagonal.h>
#include <TNL/File.h>
#include <TNL/Containers/Vector.h>

using namespace TNL;

template< typename RealType, typename Device, typename IndexType >
class TridiagonalTester : public CppUnit :: TestCase
{
   public:
   typedef Matrices::Tridiagonal< RealType, Device, IndexType > MatrixType;
   typedef Containers::Vector< RealType, Device, IndexType > VectorType;
   typedef TridiagonalTester< RealType, Device, IndexType > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;

   TridiagonalTester(){};

   virtual
   ~TridiagonalTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "TridiagonalTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "setDimensionsTest", &TesterType::setDimensionsTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setLikeTest", &TesterType::setLikeTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setElementTest", &TesterType::setElementTest ) );
      suiteOfTests -> addTest( new TestCallerType( "addElementTest", &TesterType::addElementTest ) );
      suiteOfTests -> addTest( new TestCallerType( "setRowTest", &TesterType::setRowTest ) );
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
            if( std::abs( i - j ) <= 1 )
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
         m.setElement( i, i, i );
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
            if( std::abs( i - j ) <= 1 )
               m.addElement( i, j, 1 );

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
            if( i == j )
               CPPUNIT_ASSERT( m.getElement( i, i ) == i + 1 );
            else
               if( std::abs( i - j ) == 1 )
                  CPPUNIT_ASSERT( m.getElement( i, j ) == 1 );
               else
                  CPPUNIT_ASSERT( m.getElement( i, j ) == 0 );
   }

   void setRowTest()
   {
      MatrixType m;
      m.setDimensions( 10, 10 );
      m.setValue( 0.0 );
      for( int i = 0; i < 10; i++ )
         m.setElement( i, i, i );

      Containers::Vector< IndexType, Devices::Host, IndexType > columns;
      Containers::Vector< RealType, Devices::Host, IndexType > values;
      columns.setSize( 3 );
      values.setSize( 3 );
      for( IndexType i = 4; i <= 6; i++ )
      {
         columns.setElement( i - 4, i );
         values.setElement( i - 4, i );
      }
      m.setRow( 5, columns.getData(), values.getData(), 3 );

      for( int i = 0; i < 10; i++ )
         for( int j = 0; j < 10; j++ )
         {
            if( i == 5 && j >= 4 && j <= 6 )
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
            if( std::abs( i - j ) <= 1 )
               m.setElement( i, j, i*size + j );

      MatrixType m2;
      m2.setLike( m );
      m2.setValue( 3.0 );
      m2.addMatrix( m );

      for( int i = 0; i < size; i++ )
         for( int j = 0; j < size; j++ )
            if( std::abs( i - j ) <= 1 )
               CPPUNIT_ASSERT( m2.getElement( i, j ) == m.getElement( i, j ) + 3.0 );
            else
               CPPUNIT_ASSERT( m2.getElement( i, j ) == 0.0 );

      m2.addMatrix( m, 0.5, 0.0 );

      for( int i = 0; i < size; i++ )
         for( int j = 0; j < size; j++ )
            if( std::abs( i - j ) <= 1 )
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
            if( std::abs( i - j ) <= 1 )
               m.setElement( i, j, i*size + j );

      MatrixType mTransposed;
      mTransposed.setLike( m );
      mTransposed.template getTransposition< typename MatrixType::RealType,
                                             typename MatrixType::IndexType >( m );

      for( int i = 0; i < size; i++ )
         for( int j = 0; j < size; j++ )
            CPPUNIT_ASSERT( m.getElement( i, j ) == mTransposed.getElement( j, i ) );
   }
};

#endif /* HAVE_CPPUNIT */

#endif /* TridiagonalTESTER_H_ */
