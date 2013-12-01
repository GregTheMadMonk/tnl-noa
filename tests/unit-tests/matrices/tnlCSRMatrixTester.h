/***************************************************************************
                          tnlCSRMatrixTester.h  -  description
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

#ifndef TNLCSRMATRIXTESTER_H_
#define TNLCSRMATRIXTESTER_H_

#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <matrices/tnlCSRMatrix.h>

template< class T > class tnlCSRMatrixTester : public CppUnit :: TestCase
{
   public:
   tnlCSRMatrixTester(){};

   virtual
   ~tnlCSRMatrixTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlCSRMatrixTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCSRMatrixTester< T > >(
                               "diagonalMatrixTest",
                               & tnlCSRMatrixTester< T > :: diagonalMatrixTest )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCSRMatrixTester< T > >(
                               "triDiagonalMatrixTest",
                               & tnlCSRMatrixTester< T > :: triDiagonalMatrixTest )
                             );

            return suiteOfTests;
   }

   void diagonalMatrixTest()
   {
      tnlCSRMatrix< T > csr_matrix( "test-matrix:Diagonal" );
      csr_matrix. setSize( 10 );
      csr_matrix. setNonzeroElements( 10 );
      for( int i = 0; i < 10; i ++ )
         csr_matrix. setElement( i, i, T( i ) );
      csr_matrix. printOut( cout );
      bool error( false );
      for( int i = 0; i < 10; i ++ )
         if( csr_matrix. getElement( i, i ) != T( i ) )
            error = true;
      CPPUNIT_ASSERT( ! error );

      // Now try it again
      for( int i = 0; i < 10; i ++ )
         csr_matrix. setElement( i, i, T( i ) );
      csr_matrix. printOut( cout );
      error = false;
      for( int i = 0; i < 10; i ++ )
         if( csr_matrix. getElement( i, i ) != T( i ) )
            error = true;
      CPPUNIT_ASSERT( ! error );

      // Now set zeros on the diagonal
      for( int i = 0; i < 10; i ++ )
         csr_matrix. setElement( i, i, T( 0.0 ) );
      csr_matrix. printOut( cout );
      error = false;
      for( int i = 0; i < 10; i ++ )
         if( csr_matrix. getElement( i, i ) != T( 0.0 ) )
            error = true;
      CPPUNIT_ASSERT( ! error );

      // Now again but backward
      for( int i = 9; i >= 0; i -- )
         csr_matrix. setElement( i, i, T( i ) );
      csr_matrix. printOut( cout );
      error = false;
      for( int i = 0; i < 10; i ++ )
         if( csr_matrix. getElement( i, i ) != T( i ) )
            error = true;
      CPPUNIT_ASSERT( ! error );


   };

   void triDiagonalMatrixTest()
   {
      tnlCSRMatrix< T > csr_matrix( "test-matrix:Tridiagonal" );
      csr_matrix. setSize( 10 );
      csr_matrix. setNonzeroElements( 30 );
      T data[] = { -1.0, 2.0, -1.0 };
      int offsets[] = { -1, 0, 1 };
      for( int i = 0; i < 10; i ++ )
      {
         csr_matrix. insertRow( i,      // row
                                3,      // elements
                                data,   // data
                                i,      // first column
                                offsets );
      }
      csr_matrix. printOut( cout );
      bool error( false );
      for( int i = 0; i < 10; i ++ )
      {
         if( csr_matrix. getElement( i, i ) != T( 2.0 ) )
            error = true;
         if( i > 0 && csr_matrix. getElement( i, i - 1 ) != T( -1.0 ) )
            error = true;
         if( i < 9 && csr_matrix. getElement( i, i + 1 ) != T( -1.0 ) )
            error = true;
      }
      CPPUNIT_ASSERT( ! error );

      // Backward
      tnlCSRMatrix< T > csr_matrix2( "test-matrix:Tridiagonal" );
      csr_matrix2. setSize( 10 );
      csr_matrix2. setNonzeroElements( 30 );
      for( int i = 9; i >= 0; i -- )
      {
         csr_matrix2. insertRow( i,      // row
                                3,      // elements
                                data,   // data
                                i,      // first column
                                offsets );
      }
      csr_matrix2. printOut( cout );
      error = false;
      for( int i = 0; i < 10; i ++ )
      {
         if( csr_matrix2. getElement( i, i ) != T( 2.0 ) )
            error = true;
         if( i > 0 && csr_matrix2. getElement( i, i - 1 ) != T( -1.0 ) )
            error = true;
         if( i < 9 && csr_matrix2. getElement( i, i + 1 ) != T( -1.0 ) )
            error = true;
      }
      CPPUNIT_ASSERT( ! error );

   }
};

#endif /* TNLCSRMATRIXTESTER_H_ */
