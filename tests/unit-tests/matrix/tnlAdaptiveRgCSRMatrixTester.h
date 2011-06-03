/***************************************************************************
                          tnlAdaptiveRgCSRMatrixTester.h  -  description
                             -------------------
    begin                : Jul 20, 2010
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

#ifndef TNLAdaptiveRgCSRMATRIXTESTER_H_
#define TNLAdaptiveRgCSRMATRIXTESTER_H_

#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <matrix/tnlCSRMatrix.h>
#include <matrix/tnlMatrixTester.h>
#include <matrix/tnlAdaptiveRgCSRMatrix.h>

template< class Real, tnlDevice Device > class tnlAdaptiveRgCSRMatrixTester : public CppUnit :: TestCase,
                                                                           public tnlMatrixTester< Real >
{
   public:
   tnlAdaptiveRgCSRMatrixTester(){};

   virtual
   ~tnlAdaptiveRgCSRMatrixTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlAdaptiveRgCSRMatrixTester" );
      CppUnit :: TestResult result;

      if( Device == tnlHost )
      {
         suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< Real, Device > >(
                                  "ifEmptyMatrixIsStoredProperly",
                                  & tnlAdaptiveRgCSRMatrixTester< Real, Device > :: ifEmptyMatrixIsStoredProperly )
                                );
         suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< Real, Device > >(
                                  "ifDiagonalMatrixIsStoredProperly",
                                  & tnlAdaptiveRgCSRMatrixTester< Real, Device > :: ifDiagonalMatrixIsStoredProperly )
                                );
         suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< Real, Device > >(
                                  "ifTriDiagonalMatrixIsStoredProperly",
                                  & tnlAdaptiveRgCSRMatrixTester< Real, Device > :: ifTriDiagonalMatrixIsStoredProperly )
                                );
         suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< Real, Device > >(
                                  "ifUpperTriangularMatrixIsStoredProperly",
                                  & tnlAdaptiveRgCSRMatrixTester< Real, Device > :: ifUpperTriangularMatrixIsStoredProperly )
                                );
         suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< Real, Device > >(
                                  "ifFullMatrixIsStoredProperly",
                                  & tnlAdaptiveRgCSRMatrixTester< Real, Device > :: ifFullMatrixIsStoredProperly )
                                );
         suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< Real, Device > >(
                                  "ifBcsstk20MatrixIsStoredProperly",
                                  & tnlAdaptiveRgCSRMatrixTester< Real, Device > :: ifBcsstk20MatrixIsStoredProperly )
                                );
      }

      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< Real, Device > >(
                               "ifSpmvWithEmptyMatrixWorks",
                               & tnlAdaptiveRgCSRMatrixTester< Real, Device > :: ifSpmvWithEmptyMatrixWorks )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< Real, Device > >(
                               "ifSpmvWithDiagonalMatrixWorks",
                               & tnlAdaptiveRgCSRMatrixTester< Real, Device > :: ifSpmvWithDiagonalMatrixWorks )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< Real, Device > >(
                               "ifSpmvWithTriDiagonalMatrixWorks",
                               & tnlAdaptiveRgCSRMatrixTester< Real, Device > :: ifSpmvWithTriDiagonalMatrixWorks )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< Real, Device > >(
                               "ifSpmvWithUpperTriangularMatrixWorks",
                               & tnlAdaptiveRgCSRMatrixTester< Real, Device > :: ifSpmvWithUpperTriangularMatrixWorks  )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< Real, Device > >(
                               "ifSpmvWithFullMatrixWorks",
                               & tnlAdaptiveRgCSRMatrixTester< Real, Device > :: ifSpmvWithFullMatrixWorks  )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< Real, Device > >(
                               "ifSpmvWithBcsstk20MatrixWorks",
                               & tnlAdaptiveRgCSRMatrixTester< Real, Device > :: ifSpmvWithBcsstk20MatrixWorks )
                             );
      return suiteOfTests;
   }

   void ifEmptyMatrixIsStoredProperly()
   {
      const int size = 12;
      tnlCSRMatrix< Real > csrMatrix( "test-matrix:Empty" );
      tnlAdaptiveRgCSRMatrix< Real, Device > argcsrMatrix( "test-matrix:Empty" );
      setEmptyMatrix( csrMatrix, size );
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
      tnlCSRMatrix< Real > csrMatrix( "test-matrix:Diagonal" );
      tnlAdaptiveRgCSRMatrix< Real, Device > argcsrMatrix( "test-matrix:Diagonal" );
      setDiagonalMatrix( csrMatrix, size );
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
      tnlCSRMatrix< Real > csrMatrix( "test-matrix:Tridiagonal" );
      tnlAdaptiveRgCSRMatrix< Real, Device > argcsrMatrix( "test-matrix:Tridiagonal" );
      int size = 12;
      setTridiagonalMatrix( csrMatrix, size );
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
      tnlCSRMatrix< Real > csrMatrix( "test-matrix:upperTriangular" );
      tnlAdaptiveRgCSRMatrix< Real, Device > argcsrMatrix( "test-matrix:upperTriangular" );
      const int size = 12;
      setUpperTriangularMatrix( csrMatrix, size );
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
      tnlCSRMatrix< Real > csrMatrix( "test-matrix:full" );
      tnlAdaptiveRgCSRMatrix< Real, Device > argcsrMatrix( "test-matrix:full" );
      const int size = 12;
      setFullMatrix( csrMatrix, size );
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
      tnlCSRMatrix< Real > csrMatrix( "test-matrix:bcsstk20" );
      tnlAdaptiveRgCSRMatrix< Real, Device > argcsrMatrix( "test-matrix:bcsstk20" );
      const int size = 12;
      setBcsstk20Matrix( csrMatrix );
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
      tnlCSRMatrix< Real > csrMatrix( "test-matrix:Empty" );
      tnlAdaptiveRgCSRMatrix< Real, Device > argcsrMatrix( "test-matrix:Empty" );
      setEmptyMatrix( csrMatrix, size );
      if( Device == tnlHost )
         argcsrMatrix. copyFrom( csrMatrix );
      else
      {
         tnlAdaptiveRgCSRMatrix< Real, tnlHost > hostArgcsrMatrix( "test-matrix:host-aux" );
         hostArgcsrMatrix. copyFrom( csrMatrix );
         argcsrMatrix. copyFrom( hostArgcsrMatrix );
      }

      tnlLongVector< Real, tnlHost > x1( "x1" ), b1( "b1" );
      tnlLongVector< Real, Device > x2( "x2" ), b2( "b2" );
      x1. setSize( size );
      x2. setSize( size );
      b1. setSize( size );
      b2. setSize( size );
      x1. setValue( 1.0 );
      x2. setValue( 1.0 );
      csrMatrix. vectorProduct( x1, b1 );
      argcsrMatrix. vectorProduct( x2, b2 );

      CPPUNIT_ASSERT( b1 == b2 );
   }


   void ifSpmvWithDiagonalMatrixWorks()
   {
      const int size = 35;
      tnlCSRMatrix< Real > csrMatrix( "test-matrix:Diagonal" );
      tnlAdaptiveRgCSRMatrix< Real, Device > argcsrMatrix( "test-matrix:Diagonal" );
      setDiagonalMatrix( csrMatrix, size );
      if( Device == tnlHost )
         argcsrMatrix. copyFrom( csrMatrix );
      else
      {
         tnlAdaptiveRgCSRMatrix< Real, tnlHost > hostArgcsrMatrix( "test-matrix:host-aux" );
         hostArgcsrMatrix. copyFrom( csrMatrix );
         argcsrMatrix. copyFrom( hostArgcsrMatrix );
      }

      tnlLongVector< Real, tnlHost > x1( "x1" ), b1( "b1" );
      tnlLongVector< Real, Device > x2( "x2" ), b2( "b2" );
      x1. setSize( size );
      x2. setSize( size );
      b1. setSize( size );
      b2. setSize( size );
      x1. setValue( 1.0 );
      x2. setValue( 1.0 );
      csrMatrix. vectorProduct( x1, b1 );
      argcsrMatrix. vectorProduct( x2, b2 );

      CPPUNIT_ASSERT( b1 == b2 );
   }

   void ifSpmvWithTriDiagonalMatrixWorks()
   {
      const int size = 12;
      tnlCSRMatrix< Real > csrMatrix( "test-matrix:TriDiagonal" );
      tnlAdaptiveRgCSRMatrix< Real, Device > argcsrMatrix( "test-matrix:TriDiagonal" );
      setTridiagonalMatrix( csrMatrix, size );
      if( Device == tnlHost )
         argcsrMatrix. copyFrom( csrMatrix );
      else
      {
         tnlAdaptiveRgCSRMatrix< Real, tnlHost > hostArgcsrMatrix( "test-matrix:host-aux" );
         hostArgcsrMatrix. copyFrom( csrMatrix );
         argcsrMatrix. copyFrom( hostArgcsrMatrix );
      }

      tnlLongVector< Real, tnlHost > x1( "x1" ), b1( "b1" );
      tnlLongVector< Real, Device > x2( "x2" ), b2( "b2" );
      x1. setSize( size );
      x2. setSize( size );
      b1. setSize( size );
      b2. setSize( size );
      x1. setValue( 1.0 );
      x2. setValue( 1.0 );
      csrMatrix. vectorProduct( x1, b1 );
      argcsrMatrix. vectorProduct( x2, b2 );
      CPPUNIT_ASSERT( b1 == b2 );
   }

   void ifSpmvWithUpperTriangularMatrixWorks()
   {
      const int size = 12;
      tnlCSRMatrix< Real > csrMatrix( "test-matrix:TriDiagonal" );
      tnlAdaptiveRgCSRMatrix< Real, Device > argcsrMatrix( "test-matrix:TriDiagonal" );
      setUpperTriangularMatrix( csrMatrix, size );
      if( Device == tnlHost )
         argcsrMatrix. copyFrom( csrMatrix );
      else
      {
         tnlAdaptiveRgCSRMatrix< Real, tnlHost > hostArgcsrMatrix( "test-matrix:host-aux" );
         hostArgcsrMatrix. copyFrom( csrMatrix );
         argcsrMatrix. copyFrom( hostArgcsrMatrix );
      }

      tnlLongVector< Real, tnlHost > x1( "x1" ), b1( "b1" );
      tnlLongVector< Real, Device > x2( "x2" ), b2( "b2" );
      x1. setSize( size );
      x2. setSize( size );
      b1. setSize( size );
      b2. setSize( size );
      x1. setValue( 1.0 );
      x2. setValue( 1.0 );
      csrMatrix. vectorProduct( x1, b1 );
      argcsrMatrix. vectorProduct( x2, b2 );

      CPPUNIT_ASSERT( b1 == b2 );
   }

   void ifSpmvWithFullMatrixWorks()
   {
      const int size = 12;
      tnlCSRMatrix< Real > csrMatrix( "test-matrix:full" );
      tnlAdaptiveRgCSRMatrix< Real, Device > argcsrMatrix( "test-matrix:full" );
      setFullMatrix( csrMatrix, size );
      if( Device == tnlHost )
         argcsrMatrix. copyFrom( csrMatrix );
      else
      {
         tnlAdaptiveRgCSRMatrix< Real, tnlHost > hostArgcsrMatrix( "test-matrix:host-aux" );
         hostArgcsrMatrix. copyFrom( csrMatrix );
         argcsrMatrix. copyFrom( hostArgcsrMatrix );
      }

      tnlLongVector< Real, tnlHost > x1( "x1" ), b1( "b1" );
      tnlLongVector< Real, Device > x2( "x2" ), b2( "b2" );
      x1. setSize( size );
      x2. setSize( size );
      b1. setSize( size );
      b2. setSize( size );
      x1. setValue( 1.0 );
      x2. setValue( 1.0 );
      csrMatrix. vectorProduct( x1, b1 );
      argcsrMatrix. vectorProduct( x2, b2 );

      CPPUNIT_ASSERT( b1 == b2 );
   }

   void ifSpmvWithBcsstk20MatrixWorks()
   {
      tnlCSRMatrix< Real > csrMatrix( "test-matrix:TriDiagonal" );
      tnlAdaptiveRgCSRMatrix< Real, Device > argcsrMatrix( "test-matrix:TriDiagonal" );
      setBcsstk20Matrix( csrMatrix );
      if( Device == tnlHost )
         argcsrMatrix. copyFrom( csrMatrix );
      else
      {
         tnlAdaptiveRgCSRMatrix< Real, tnlHost > hostArgcsrMatrix( "test-matrix:host-aux" );
         hostArgcsrMatrix. copyFrom( csrMatrix );
         argcsrMatrix. copyFrom( hostArgcsrMatrix );
      }

      const int size = csrMatrix. getSize();
      tnlLongVector< Real, tnlHost > x1( "x1" ), b1( "b1" );
      tnlLongVector< Real, Device > x2( "x2" ), b2( "b2" );
      x1. setSize( size );
      x2. setSize( size );
      b1. setSize( size );
      b2. setSize( size );
      x1. setValue( 1.0 );
      x2. setValue( 1.0 );
      csrMatrix. vectorProduct( x1, b1 );
      argcsrMatrix. vectorProduct( x2, b2 );

      Real maxError( 0.0 );
      for( int j = 0; j < b1. getSize(); j ++ )
      {
         //f << refB[ j ] << " - " << host_b[ j ] << " = "  << refB[ j ] - host_b[ j ] <<  endl;
         Real error( 0.0 );
         if( b1. getElement( j ) != 0.0 )
            error = ( Real ) fabs( b1. getElement( j ) - b2. getElement( j ) ) /  ( Real ) fabs( b1. getElement( j ) );
         else
            error = ( Real ) fabs( b2. getElement( j ) );
         maxError = Max( maxError, error );
      }
      CPPUNIT_ASSERT( maxError < 1.0e-12 );
   }
};

#endif /* TNLAdaptiveRgCSRMATRIXTESTER_H_ */
