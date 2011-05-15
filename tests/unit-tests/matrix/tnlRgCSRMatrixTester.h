/***************************************************************************
                          tnlRgCSRMatrixTester.h  -  description
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

#ifndef TNLRgCSRMATRIXTESTER_H_
#define TNLRgCSRMATRIXTESTER_H_

#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <matrix/tnlCSRMatrix.h>
#include <matrix/tnlRgCSRMatrix.h>

template< class T > class tnlRgCSRMatrixTester : public CppUnit :: TestCase
{
   public:
   tnlRgCSRMatrixTester(){};

   virtual
   ~tnlRgCSRMatrixTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlRgCSRMatrixTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlRgCSRMatrixTester< T > >(
                               "diagonalMatrixTest",
                               & tnlRgCSRMatrixTester< T > :: diagonalMatrixTest )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlRgCSRMatrixTester< T > >(
                               "triDiagonalMatrixTest",
                               & tnlRgCSRMatrixTester< T > :: triDiagonalMatrixTest )
                             );

            return suiteOfTests;
   }

   void diagonalMatrixTest()
   {
      tnlCSRMatrix< T > csr_matrix( "test-matrix:Diagonal" );
      tnlRgCSRMatrix< T > coacsr_matrix( "test-matrix:Diagonal" );
      csr_matrix. setSize( 12 );
      csr_matrix. setNonzeroElements( 12 );
      for( int i = 0; i < 12; i ++ )
         csr_matrix. setElement( i, i, T( i + 1 ) );
      //cerr << "Copying data to coalesced CSR matrix." << endl;
      coacsr_matrix. copyFrom( csr_matrix, 4 );
      //coacsr_matrix. printOut( cout );
      bool error( false );
      for( int i = 0; i < 12; i ++ )
         if( coacsr_matrix. getElement( i, i ) != T( i + 1 ) )
            error = true;
      CPPUNIT_ASSERT( ! error );
   };

   void triDiagonalMatrixTest()
   {
      tnlCSRMatrix< T > csr_matrix( "test-matrix:Tridiagonal" );
      tnlRgCSRMatrix< T > coacsr_matrix( "test-matrix:Tridiagonal" );
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
      coacsr_matrix. copyFrom( csr_matrix, 4 );
      //cerr << "----------------" << endl;
      //coacsr_matrix. printOut( cout );
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

#endif /* TNLRgCSRMATRIXTESTER_H_ */
