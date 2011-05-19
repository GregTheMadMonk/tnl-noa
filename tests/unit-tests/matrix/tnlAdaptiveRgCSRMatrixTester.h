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
#include <matrix/tnlAdaptiveRgCSRMatrix.h>

template< class T > class tnlAdaptiveRgCSRMatrixTester : public CppUnit :: TestCase
{
   public:
   tnlAdaptiveRgCSRMatrixTester(){};

   virtual
   ~tnlAdaptiveRgCSRMatrixTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlAdaptiveRgCSRMatrixTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< T > >(
                               "diagonalMatrixTest",
                               & tnlAdaptiveRgCSRMatrixTester< T > :: diagonalMatrixTest )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< T > >(
                               "triDiagonalMatrixTest",
                               & tnlAdaptiveRgCSRMatrixTester< T > :: triDiagonalMatrixTest )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< T > >(
                               "upperTriangularMatrixTest",
                               & tnlAdaptiveRgCSRMatrixTester< T > :: upperTriangularMatrixTest )
                             );


            return suiteOfTests;
   }

   void diagonalMatrixTest()
   {
      tnlCSRMatrix< T > csr_matrix( "test-matrix:Diagonal" );
      tnlAdaptiveRgCSRMatrix< T > argcsrMatrix( "test-matrix:Diagonal" );
      csr_matrix. setSize( 12 );
      csr_matrix. setNonzeroElements( 12 );
      for( int i = 0; i < 12; i ++ )
         csr_matrix. setElement( i, i, T( i + 1 ) );
      //cerr << "Copying data to coalesced CSR matrix." << endl;
      argcsrMatrix. copyFrom( csr_matrix );
      //argcsrMatrix. printOut( cout );
      bool error( false );
      for( int i = 0; i < 12; i ++ )
         if( argcsrMatrix. getElement( i, i ) != T( i + 1 ) )
            error = true;
      CPPUNIT_ASSERT( ! error );
   };

   void triDiagonalMatrixTest()
   {
      tnlCSRMatrix< T > csr_matrix( "test-matrix:Tridiagonal" );
      tnlAdaptiveRgCSRMatrix< T > argcsr_matrix( "test-matrix:Tridiagonal" );
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
      argcsr_matrix. copyFrom( csr_matrix );
      //argcsr_matrix. printOut( cout );
      bool error( false );
      for( int i = 0; i < size; i ++ )
         for( int j = 0; j < size; j ++ )
            if( csr_matrix. getElement( i, j ) != argcsr_matrix. getElement( i, j ) )
               error = true;
      CPPUNIT_ASSERT( ! error );

   }
   void upperTriangularMatrixTest()
   {
      tnlCSRMatrix< T > csr_matrix( "test-matrix:upperTriangular" );
      tnlAdaptiveRgCSRMatrix< T > argcsr_matrix( "test-matrix:upperTriangular" );
      int size = 12;
      csr_matrix. setSize( size );
      csr_matrix. setNonzeroElements( size * size );
      for( int i = 0; i < size; i ++ )
         for( int j = i; j < size; j ++ )
            csr_matrix. setElement( i, j, 1.0 );

      argcsr_matrix. copyFrom( csr_matrix );
      //cerr << "----------------" << endl;
      //argcsr_matrix. printOut( cout );
      bool error( false );
      for( int i = 0; i < size; i ++ )
         for( int j = 0; j < size; j ++ )
            if( csr_matrix. getElement( i, j ) != argcsr_matrix. getElement( i, j ) )
               error = true;

      CPPUNIT_ASSERT( ! error );

   }

};

#endif /* TNLAdaptiveRgCSRMATRIXTESTER_H_ */
