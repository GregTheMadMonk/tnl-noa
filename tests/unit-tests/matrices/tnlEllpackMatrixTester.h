/***************************************************************************
                          tnlEllpackMatrixTester.h  -  description
                             -------------------
    begin                : Jul 31, 2010
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

#ifndef TNLELLPACKMATRIXTESTER_H_
#define TNLELLPACKMATRIXTESTER_H_


#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <matrices/tnlCSRMatrix.h>
#include <matrices/tnlEllpackMatrix.h>

template< class T > class tnlEllpackMatrixTester : public CppUnit :: TestCase
{
   public:
   tnlEllpackMatrixTester(){};

   virtual
   ~tnlEllpackMatrixTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlEllpackMatrixTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlEllpackMatrixTester< T > >(
                               "diagonalMatrixTest",
                               & tnlEllpackMatrixTester< T > :: diagonalMatrixTest )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlEllpackMatrixTester< T > >(
                               "triDiagonalMatrixTest",
                               & tnlEllpackMatrixTester< T > :: triDiagonalMatrixTest )
                             );

            return suiteOfTests;
   }

   void diagonalMatrixTest()
   {
      tnlCSRMatrix< T > csr_matrix( "test-matrix:Diagonal" );
      tnlEllpackMatrix< T > ellpack_matrix( "test-matrix:Diagonal", 4 );
      csr_matrix. setSize( 12 );
      csr_matrix. setNonzeroElements( 12 );
      for( int i = 0; i < 12; i ++ )
         csr_matrix. setElement( i, i, T( i + 1 ) );
      //cerr << "Copying data to coalesced CSR matrix." << endl;
      ellpack_matrix. copyFrom( csr_matrix );
      //ellpack_matrix. printOut( cout );
      bool error( false );
      for( int i = 0; i < 12; i ++ )
         if( ellpack_matrix. getElement( i, i ) != T( i + 1 ) )
            error = true;
      //cout << ellpack_matrix << endl;
      CPPUNIT_ASSERT( ! error );
   };

   void triDiagonalMatrixTest()
   {
      tnlCSRMatrix< T > csr_matrix( "test-matrix:Tridiagonal" );
      tnlEllpackMatrix< T > ellpack_matrix( "test-matrix:Tridiagonal", 4 );
      int size = 12;
      csr_matrix. setSize( size );
      csr_matrix. setNonzeroElements( size * 3 - 2 );
      T data[] = { -1.0, 2.0, -1.0 };
      int offsets[] = { -1, 0, 1 };
      for( int i = 0; i < size; i ++ )
      {
         csr_matrix. insertRow( i,      // row
                                3,      // elements
                                data,   // data
                                i,      // first column
                                offsets );
      }
      ellpack_matrix. copyFrom( csr_matrix );
      //cerr << "----------------" << endl;
      //cout << ellpack_matrix << endl;
      bool error( false );
      for( int i = 0; i < size; i ++ )
      {
         if( csr_matrix. getElement( i, i ) != T( 2.0 ) )
            error = true;
         if( i > 0 && csr_matrix. getElement( i, i - 1 ) != T( -1.0 ) )
            error = true;
         if( i < size - 1 && csr_matrix. getElement( i, i + 1 ) != T( -1.0 ) )
            error = true;
      }
      CPPUNIT_ASSERT( ! error );

   }
};


#endif /* TNLELLPACKMATRIXTESTER_H_ */
