/***************************************************************************
                          tnlRgCSRTester.h  -  description
                             -------------------
    begin                : Jul 20, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLRgCSRMATRIXTESTER_H_
#define TNLRgCSRMATRIXTESTER_H_

#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <TNL/Matrices/CSR.h>
#include "MatrixTester.h"
#include <TNL/legacy/matrices/tnlRgCSR.h>

template< class T > class tnlRgCSRTester : public CppUnit :: TestCase,
                                                 public MatrixTester< T >
{
   public:
   tnlRgCSRTester(){};

   virtual
   ~tnlRgCSRTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlRgCSRTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlRgCSRTester< T > >(
                               "ifEmptyMatrixIsStoredProperly",
                               & tnlRgCSRTester< T > :: ifEmptyMatrixIsStoredProperly )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlRgCSRTester< T > >(
                               "ifDiagonalMatrixIsStoredProperly",
                               & tnlRgCSRTester< T > :: ifDiagonalMatrixIsStoredProperly )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlRgCSRTester< T > >(
                               "ifTriDiagonalMatrixIsStoredProperly",
                               & tnlRgCSRTester< T > :: ifTriDiagonalMatrixIsStoredProperly )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlRgCSRTester< T > >(
                               "ifSpmvWithEmptyMatrixWorks",
                               & tnlRgCSRTester< T > :: ifSpmvWithEmptyMatrixWorks )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlRgCSRTester< T > >(
                               "ifUpperTriangularMatrixIsStoredProperly",
                               & tnlRgCSRTester< T > :: ifUpperTriangularMatrixIsStoredProperly )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlRgCSRTester< T > >(
                               "ifFullMatrixIsStoredProperly",
                               & tnlRgCSRTester< T > :: ifFullMatrixIsStoredProperly )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlRgCSRTester< T > >(
                               "ifBcsstk20MatrixIsStoredProperly",
                               & tnlRgCSRTester< T > :: ifBcsstk20MatrixIsStoredProperly )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlRgCSRTester< T > >(
                               "ifSpmvWithDiagonalMatrixWorks",
                               & tnlRgCSRTester< T > :: ifSpmvWithDiagonalMatrixWorks )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlRgCSRTester< T > >(
                               "ifSpmvWithTriDiagonalMatrixWorks",
                               & tnlRgCSRTester< T > :: ifSpmvWithTriDiagonalMatrixWorks )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlRgCSRTester< T > >(
                               "ifSpmvWithUpperTriangularMatrixWorks",
                               & tnlRgCSRTester< T > :: ifSpmvWithUpperTriangularMatrixWorks  )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlRgCSRTester< T > >(
                               "ifSpmvWithFullMatrixWorks",
                               & tnlRgCSRTester< T > :: ifSpmvWithFullMatrixWorks  )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlRgCSRTester< T > >(
                               "ifSpmvWithBcsstk20MatrixWorks",
                               & tnlRgCSRTester< T > :: ifSpmvWithBcsstk20MatrixWorks )
                             );
      return suiteOfTests;
   }

   void ifEmptyMatrixIsStoredProperly()
   {
      const int size = 12;
      CSR< T > csrMatrix;
      tnlRgCSR< T > argcsrMatrix;
      this->setEmptyMatrix( csrMatrix, size );
      argcsrMatrix. copyFrom( csrMatrix );

      bool error( false );
      for( int i = 0; i < size; i ++ )
         for( int j = 0; j < size; j ++ )
            if( argcsrMatrix. getElement( i, j ) != 0 )
               error = true;
      CPPUNIT_ASSERT( ! error );
   };

   void ifDiagonalMatrixIsStoredProperly()
   {
      const int size = 12;
      CSR< T > csrMatrix;
      tnlRgCSR< T > argcsrMatrix;
      this->setDiagonalMatrix( csrMatrix, size );
      argcsrMatrix. copyFrom( csrMatrix );

      bool error( false );
      for( int i = 0; i < size; i ++ )
         for( int j = 0; j < size; j ++ )
            if( argcsrMatrix. getElement( i, j ) != csrMatrix. getElement( i, j ) )
               error = true;
      CPPUNIT_ASSERT( ! error );
   };

   void ifTriDiagonalMatrixIsStoredProperly()
   {
      CSR< T > csrMatrix;
      tnlRgCSR< T > argcsrMatrix;
      int size = 12;
      this->setTridiagonal( csrMatrix, size );
      argcsrMatrix. copyFrom( csrMatrix );

      bool error( false );
      for( int i = 0; i < size; i ++ )
         for( int j = 0; j < size; j ++ )
            if( csrMatrix. getElement( i, j ) != argcsrMatrix. getElement( i, j ) )
               error = true;
      CPPUNIT_ASSERT( ! error );

   }

   void ifUpperTriangularMatrixIsStoredProperly()
   {
      CSR< T > csrMatrix;
      tnlRgCSR< T > argcsrMatrix;
      const int size = 12;
      this->setUpperTriangularMatrix( csrMatrix, size );
      argcsrMatrix. copyFrom( csrMatrix );

      bool error( false );
      for( int i = 0; i < size; i ++ )
         for( int j = 0; j < size; j ++ )
            if( csrMatrix. getElement( i, j ) != argcsrMatrix. getElement( i, j ) )
               error = true;

      CPPUNIT_ASSERT( ! error );
   }

   void ifFullMatrixIsStoredProperly()
   {
      CSR< T > csrMatrix;
      tnlRgCSR< T > argcsrMatrix;
      const int size = 12;
      this->setUpperTriangularMatrix( csrMatrix, size );
      argcsrMatrix. copyFrom( csrMatrix );

      bool error( false );
      for( int i = 0; i < size; i ++ )
         for( int j = 0; j < size; j ++ )
            if( csrMatrix. getElement( i, j ) != argcsrMatrix. getElement( i, j ) )
               error = true;

      CPPUNIT_ASSERT( ! error );
   }

   void ifBcsstk20MatrixIsStoredProperly()
   {
      CSR< T > csrMatrix;
      tnlRgCSR< T > argcsrMatrix;
      const int size = 12;
      this->setBcsstk20Matrix( csrMatrix );
      argcsrMatrix. copyFrom( csrMatrix );

      bool error( false );
      for( int i = 0; i < size; i ++ )
         for( int j = 0; j < size; j ++ )
            if( csrMatrix. getElement( i, j ) != argcsrMatrix. getElement( i, j ) )
               error = true;

      CPPUNIT_ASSERT( ! error );
   }

   void ifSpmvWithEmptyMatrixWorks()
   {
      const int size = 35;
      CSR< T > csrMatrix;
      tnlRgCSR< T > argcsrMatrix;
      this->setEmptyMatrix( csrMatrix, size );
      argcsrMatrix. copyFrom( csrMatrix );

      Vector< T > x, b1, b2;
      x. setSize( size );
      b1. setSize( size );
      b2. setSize( size );
      x. setValue( 1.0 );
      csrMatrix. vectorProduct( x, b1 );
      argcsrMatrix. vectorProduct( x, b2 );

      CPPUNIT_ASSERT( b1 == b2 );
   }

   void ifSpmvWithDiagonalMatrixWorks()
   {
      const int size = 35;
      CSR< T > csrMatrix;
      tnlRgCSR< T > argcsrMatrix;
      this->setDiagonalMatrix( csrMatrix, size );
      argcsrMatrix. copyFrom( csrMatrix );

      Vector< T > x, b1, b2;
      x. setSize( size );
      b1. setSize( size );
      b2. setSize( size );
      x. setValue( 1.0 );
      csrMatrix. vectorProduct( x, b1 );
      argcsrMatrix. vectorProduct( x, b2 );

      CPPUNIT_ASSERT( b1 == b2 );
   }

   void ifSpmvWithTriDiagonalMatrixWorks()
   {
      const int size = 12;
      CSR< T > csrMatrix;
      tnlRgCSR< T > argcsrMatrix;
      this->setTridiagonal( csrMatrix, size );
      argcsrMatrix. copyFrom( csrMatrix );

      Vector< T > x, b1, b2;
      x. setSize( size );
      b1. setSize( size );
      b2. setSize( size );
      x. setValue( 1.0 );
      csrMatrix. vectorProduct( x, b1 );
      argcsrMatrix. vectorProduct( x, b2 );

      CPPUNIT_ASSERT( b1 == b2 );
   }

   void ifSpmvWithUpperTriangularMatrixWorks()
   {
      const int size = 12;
      CSR< T > csrMatrix;
      tnlRgCSR< T > argcsrMatrix;
      this->setUpperTriangularMatrix( csrMatrix, size );
      argcsrMatrix. copyFrom( csrMatrix );

      Vector< T > x, b1, b2;
      x. setSize( size );
      b1. setSize( size );
      b2. setSize( size );
      x. setValue( 1.0 );
      csrMatrix. vectorProduct( x, b1 );
      argcsrMatrix. vectorProduct( x, b2 );

      CPPUNIT_ASSERT( b1 == b2 );
   }

   void ifSpmvWithFullMatrixWorks()
   {
      const int size = 12;
      CSR< T > csrMatrix;
      tnlRgCSR< T > argcsrMatrix;
      this->setFullMatrix( csrMatrix, size );
      argcsrMatrix. copyFrom( csrMatrix );

      Vector< T > x, b1, b2;
      x. setSize( size );
      b1. setSize( size );
      b2. setSize( size );
      x. setValue( 1.0 );
      csrMatrix. vectorProduct( x, b1 );
      argcsrMatrix. vectorProduct( x, b2 );

      CPPUNIT_ASSERT( b1 == b2 );
   }

   void ifSpmvWithBcsstk20MatrixWorks()
   {
      CSR< T > csrMatrix;
      tnlRgCSR< T > argcsrMatrix;
      this->setBcsstk20Matrix( csrMatrix );
      argcsrMatrix. copyFrom( csrMatrix );
      const int size = csrMatrix. getRows();

      Vector< T > x( "x" ), b1( "b1" ), b2( "b2" );
      x. setSize( size );
      b1. setSize( size );
      b2. setSize( size );
      x. setValue( 1.0 );
      csrMatrix. vectorProduct( x, b1 );
      argcsrMatrix. vectorProduct( x, b2 );



      CPPUNIT_ASSERT( b1 == b2 );
   }

};

#endif /* TNLRgCSRMATRIXTESTER_H_ */
