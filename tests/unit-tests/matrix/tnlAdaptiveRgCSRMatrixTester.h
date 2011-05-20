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
                               "ifDiagonalMatrixIsStoredProperly",
                               & tnlAdaptiveRgCSRMatrixTester< T > :: ifDiagonalMatrixIsStoredProperly )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< T > >(
                               "ifTriDiagonalMatrixIsStoredProperly",
                               & tnlAdaptiveRgCSRMatrixTester< T > :: ifTriDiagonalMatrixIsStoredProperly )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< T > >(
                               "ifUpperTriangularMatrixIsStoredProperly",
                               & tnlAdaptiveRgCSRMatrixTester< T > :: ifUpperTriangularMatrixIsStoredProperly )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< T > >(
                               "ifSpmvWithDiagonalMatrixWorks",
                               & tnlAdaptiveRgCSRMatrixTester< T > :: ifSpmvWithDiagonalMatrixWorks )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< T > >(
                               "ifSpmvWithTriDiagonalMatrixWorks",
                               & tnlAdaptiveRgCSRMatrixTester< T > :: ifSpmvWithTriDiagonalMatrixWorks )
                             );

      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< T > >(
                               "ifSpmvWithUpperTriangularMatrixWorks",
                               & tnlAdaptiveRgCSRMatrixTester< T > :: ifSpmvWithUpperTriangularMatrixWorks  )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRMatrixTester< T > >(
                               "ifSpmvWithBcsstk20MatrixWorks",
                               & tnlAdaptiveRgCSRMatrixTester< T > :: ifSpmvWithBcsstk20MatrixWorks )
                             );



      return suiteOfTests;
   }

   void setDiagonalMatrix( tnlCSRMatrix< T >& csrMatrix,
                           tnlAdaptiveRgCSRMatrix< T >& argcsrMatrix,
                           const int size )
   {
      csrMatrix. setSize( size );
      csrMatrix. setNonzeroElements( size );
      for( int i = 0; i < size; i ++ )
         csrMatrix. setElement( i, i, T( i + 1 ) );
      //cerr << "Copying data to coalesced CSR matrix." << endl;
      argcsrMatrix. copyFrom( csrMatrix );
      //argcsrMatrix. printOut( cout );
   }

   void setTridiagonalMatrix( tnlCSRMatrix< T >& csrMatrix,
                              tnlAdaptiveRgCSRMatrix< T >& argcsrMatrix,
                              const int size )
   {
      csrMatrix. setSize( size );
      csrMatrix. setNonzeroElements( size * 3 - 2 );
      T data[] = { -1.0, 2.0, -1.0 };
      int offsets[] = { -1, 0, 1 };
      for( int i = 0; i < size; i ++ )
      {
         csrMatrix. insertRow( i,      // row
                               3,      // elements
                               data,   // data
                               i,      // first column
                               offsets );
      }
      argcsrMatrix. copyFrom( csrMatrix );
      //argcsr_matrix. printOut( cout );
   }

   void setUpperTriangularMatrix( tnlCSRMatrix< T >& csrMatrix,
                                  tnlAdaptiveRgCSRMatrix< T >& argcsrMatrix,
                                  const int size )
   {
      csrMatrix. setSize( size );
      csrMatrix. setNonzeroElements( size * size );
      for( int i = 0; i < size; i ++ )
         for( int j = i; j < size; j ++ )
            csrMatrix. setElement( i, j, 1.0 );

      argcsrMatrix. copyFrom( csrMatrix );
      //argcsr_matrix. printOut( cout );
   }

   void setBcsstk20Matrix( tnlCSRMatrix< T >& csrMatrix,
                           tnlAdaptiveRgCSRMatrix< T >& argcsrMatrix )
   {
      /****
       * Sets a real matrix from a matrix market
       */
      csrMatrix. setSize( 50 );
      csrMatrix. setNonzeroElements( 100 );
      
      csrMatrix. setElement( 1, 1,  3.4651842369950e+08    );
      csrMatrix. setElement( 3, 1,  3.0695529658430e+10    );
      csrMatrix. setElement( 4, 1, -3.4651842369950e+08    );
      csrMatrix. setElement( 6, 1,  3.0695529658430e+10    );
      csrMatrix. setElement( 2, 2,  5.4150467497750e+08    );
      csrMatrix. setElement( 5, 2, -5.4150467497750e+08    );
      csrMatrix. setElement( 3, 3,  3.6254562588700e+12    );
      csrMatrix. setElement( 4, 3, -3.0695529658430e+10    );
      csrMatrix. setElement( 6, 3,  1.8127281294350e+12    );
      csrMatrix. setElement( 4, 4,  5.8200610488580e+09    );
      csrMatrix. setElement( 6, 4,  1.6324889406770e+11    );
      csrMatrix. setElement( 7, 4, -5.4735426251580e+09    );
      csrMatrix. setElement( 9, 4,  1.9394442372610e+11    );
      csrMatrix. setElement( 5, 5,  1.7185252532270e+09    );
      csrMatrix. setElement( 8, 5, -1.1770205782500e+09    );
      csrMatrix. setElement( 6, 6,  1.2788184938840e+13    );
      csrMatrix. setElement( 7, 6, -1.9394442372610e+11    );
      csrMatrix. setElement( 9, 6,  4.5813643399870e+12    );
      csrMatrix. setElement( 7, 7,  1.1034132683920e+10    );
      csrMatrix. setElement( 9, 7,  3.0843578836850e+09    );
      csrMatrix. setElement( 10, 7, -5.5605900587650e+09   );
      csrMatrix. setElement( 12, 7,  1.9702878160980e+11   );
      csrMatrix. setElement( 8, 8,  2.3681428407440e+09    );
      csrMatrix. setElement( 11, 8, -1.1911222624940e+09   );
      csrMatrix. setElement( 9, 9,  1.8471175055240e+13    );
      csrMatrix. setElement( 10, 9, -1.9702878160980e+11   );
      csrMatrix. setElement( 12, 9,  4.6542231876330e+12   );
      csrMatrix. setElement( 10, 10,  1.1211709448480e+10  );
      csrMatrix. setElement( 12, 10,  3.2077321990330e+09  );
      csrMatrix. setElement( 13, 10, -5.6511193897150e+09  );
      csrMatrix. setElement( 15, 10,  2.0023651380880e+11  );
      csrMatrix. setElement( 11, 11,  2.3954060969490e+09  );
      csrMatrix. setElement( 14, 11, -1.2042838344550e+09  );
      csrMatrix. setElement( 12, 12,  1.8768439153640e+13  );
      csrMatrix. setElement( 13, 12, -2.0023651380880e+11  );
      csrMatrix. setElement( 15, 12,  4.7299963891850e+12  );
      csrMatrix. setElement( 13, 13,  1.1392768110380e+10  );
      csrMatrix. setElement( 15, 13,  3.2077321990300e+09  );
      csrMatrix. setElement( 16, 13, -5.7416487206660e+09  );
      csrMatrix. setElement( 18, 13,  2.0344424600780e+11  );
      csrMatrix. setElement( 14, 14,  2.4226693531550e+09  );
      csrMatrix. setElement( 17, 14, -1.2183855187000e+09  );
      csrMatrix. setElement( 15, 15,  1.9071531959840e+13  );
      csrMatrix. setElement( 16, 15, -2.0344424600780e+11  );
      csrMatrix. setElement( 18, 15,  4.8057695907370e+12  );
      csrMatrix. setElement( 16, 16,  1.1570344874940e+10  );
      csrMatrix. setElement( 18, 16,  3.0843578836840e+09  );
      csrMatrix. setElement( 19, 16, -5.8286961542720e+09  );
      csrMatrix. setElement( 21, 16,  2.0652860389150e+11  );
      csrMatrix. setElement( 17, 17,  2.4499326093600e+09  );
      csrMatrix. setElement( 20, 17, -1.2315470906610e+09  );
      csrMatrix. setElement( 18, 18,  1.9368796058240e+13  );
      csrMatrix. setElement( 19, 18, -2.0652860389150e+11  );
      csrMatrix. setElement( 21, 18,  4.8786284383830e+12  );
      csrMatrix. setElement( 19, 19,  1.1747921639490e+10  );
      csrMatrix. setElement( 21, 19,  3.2077321990320e+09  );
      csrMatrix. setElement( 22, 19, -5.9192254852220e+09  );
      csrMatrix. setElement( 24, 19,  2.0973633609060e+11  );
      csrMatrix. setElement( 20, 20,  2.4771958655660e+09  );
      csrMatrix. setElement( 23, 20, -1.2456487749050e+09  );
      csrMatrix. setElement( 21, 21,  1.9666060156640e+13  );
      csrMatrix. setElement( 22, 21, -2.0973633609060e+11  );
      csrMatrix. setElement( 24, 21,  4.9544016399350e+12  );
      csrMatrix. setElement( 22, 22,  1.1928980301390e+10  );
      csrMatrix. setElement( 24, 22,  3.2077321990300e+09  );
      csrMatrix. setElement( 25, 22, -6.0097548161730e+09  );
      csrMatrix. setElement( 27, 22,  2.1294406828960e+11  );
      csrMatrix. setElement( 23, 23,  2.5044591217710e+09  );
      csrMatrix. setElement( 26, 23, -1.2588103468660e+09  );
      csrMatrix. setElement( 24, 24,  1.9969152962840e+13  );
      csrMatrix. setElement( 25, 24, -2.1294406828960e+11  );
      csrMatrix. setElement( 27, 24,  5.0301748414870e+12  );
      csrMatrix. setElement( 25, 25,  1.2106557065950e+10  );
      csrMatrix. setElement( 27, 25,  3.0843578836860e+09  );
      csrMatrix. setElement( 28, 25, -6.0968022497790e+09  );
      csrMatrix. setElement( 30, 25,  2.1602842617330e+11  );
      csrMatrix. setElement( 26, 26,  2.5317223779770e+09  );
      csrMatrix. setElement( 29, 26, -1.2729120311100e+09  );
      csrMatrix. setElement( 27, 27,  2.0266417061240e+13  );
      csrMatrix. setElement( 28, 27, -2.1602842617330e+11  );
      csrMatrix. setElement( 30, 27,  5.1030336891330e+12  );
      csrMatrix. setElement( 28, 28,  1.2284133830500e+10  );
      csrMatrix. setElement( 30, 28,  3.2077321988540e+09  );
      csrMatrix. setElement( 31, 28, -6.1873315807220e+09  );
      csrMatrix. setElement( 33, 28,  2.1923615837210e+11  );
      csrMatrix. setElement( 29, 29,  2.5589856341820e+09  );
      csrMatrix. setElement( 32, 29, -1.2860736030710e+09  );
      csrMatrix. setElement( 30, 30,  2.0563681159630e+13  );
      csrMatrix. setElement( 31, 30, -2.1923615837210e+11  );
      csrMatrix. setElement( 33, 30,  5.1788068906830e+12  );
      csrMatrix. setElement( 31, 31,  1.2465192492400e+10  );
      csrMatrix. setElement( 33, 31,  3.2077321992110e+09  );
      csrMatrix. setElement( 34, 31, -6.2778609116800e+09  );
      csrMatrix. setElement( 36, 31,  2.2244389057130e+11  );
      csrMatrix. setElement( 32, 32,  2.5862488903870e+09  );
      csrMatrix. setElement( 35, 32, -1.3001752873160e+09  );
      csrMatrix. setElement( 33, 33,  2.0866773965840e+13  );


      argcsrMatrix. copyFrom( csrMatrix );

   }

   void ifDiagonalMatrixIsStoredProperly()
   {
      const int size = 12;
      tnlCSRMatrix< T > csrMatrix( "test-matrix:Diagonal" );
      tnlAdaptiveRgCSRMatrix< T > argcsrMatrix( "test-matrix:Diagonal" );
      setDiagonalMatrix( csrMatrix, argcsrMatrix, size );

      bool error( false );
      for( int i = 0; i < size; i ++ )
         if( argcsrMatrix. getElement( i, i ) != T( i + 1 ) )
            error = true;
      CPPUNIT_ASSERT( ! error );
   };

   void ifTriDiagonalMatrixIsStoredProperly()
   {
      tnlCSRMatrix< T > csr_matrix( "test-matrix:Tridiagonal" );
      tnlAdaptiveRgCSRMatrix< T > argcsr_matrix( "test-matrix:Tridiagonal" );
      int size = 12;
      setTridiagonalMatrix( csr_matrix, argcsr_matrix, size );
      bool error( false );
      for( int i = 0; i < size; i ++ )
         for( int j = 0; j < size; j ++ )
            if( csr_matrix. getElement( i, j ) != argcsr_matrix. getElement( i, j ) )
               error = true;
      CPPUNIT_ASSERT( ! error );

   }
   void ifUpperTriangularMatrixIsStoredProperly()
   {
      tnlCSRMatrix< T > csr_matrix( "test-matrix:upperTriangular" );
      tnlAdaptiveRgCSRMatrix< T > argcsr_matrix( "test-matrix:upperTriangular" );
      const int size = 12;
      setUpperTriangularMatrix( csr_matrix, argcsr_matrix, size );
      bool error( false );
      for( int i = 0; i < size; i ++ )
         for( int j = 0; j < size; j ++ )
            if( csr_matrix. getElement( i, j ) != argcsr_matrix. getElement( i, j ) )
               error = true;

      CPPUNIT_ASSERT( ! error );
   }

   void ifSpmvWithDiagonalMatrixWorks()
   {
      const int size = 12;
      tnlCSRMatrix< T > csrMatrix( "test-matrix:Diagonal" );
      tnlAdaptiveRgCSRMatrix< T > argcsrMatrix( "test-matrix:Diagonal" );
      setDiagonalMatrix( csrMatrix, argcsrMatrix, size );

      tnlLongVector< T > x( "x" ), b1( "b1" ), b2( "b2" );
      x. setSize( size );
      b1. setSize( size );
      b2. setSize( size );
      x. setValue( 1.0 );
      csrMatrix. vectorProduct( x, b1 );
      argcsrMatrix. vectorProduct( x, b2 );

      cout << b1 << endl;
      cout << b2 << endl;

      CPPUNIT_ASSERT( b1 == b2 );
   }

   void ifSpmvWithTriDiagonalMatrixWorks()
   {
      const int size = 12;
      tnlCSRMatrix< T > csrMatrix( "test-matrix:TriDiagonal" );
      tnlAdaptiveRgCSRMatrix< T > argcsrMatrix( "test-matrix:TriDiagonal" );
      setTridiagonalMatrix( csrMatrix, argcsrMatrix, size );

      tnlLongVector< T > x( "x" ), b1( "b1" ), b2( "b2" );
      x. setSize( size );
      b1. setSize( size );
      b2. setSize( size );
      x. setValue( 1.0 );
      csrMatrix. vectorProduct( x, b1 );
      argcsrMatrix. vectorProduct( x, b2 );

      cout << b1 << endl;
      cout << b2 << endl;

      CPPUNIT_ASSERT( b1 == b2 );
   }

   void ifSpmvWithUpperTriangularMatrixWorks()
   {
      const int size = 12;
      tnlCSRMatrix< T > csrMatrix( "test-matrix:TriDiagonal" );
      tnlAdaptiveRgCSRMatrix< T > argcsrMatrix( "test-matrix:TriDiagonal" );
      setUpperTriangularMatrix( csrMatrix, argcsrMatrix, size );

      tnlLongVector< T > x( "x" ), b1( "b1" ), b2( "b2" );
      x. setSize( size );
      b1. setSize( size );
      b2. setSize( size );
      x. setValue( 1.0 );
      csrMatrix. vectorProduct( x, b1 );
      argcsrMatrix. vectorProduct( x, b2 );

      cout << b1 << endl;
      cout << b2 << endl;

      CPPUNIT_ASSERT( b1 == b2 );
   }

   void ifSpmvWithBcsstk20MatrixWorks()
   {
      tnlCSRMatrix< T > csrMatrix( "test-matrix:TriDiagonal" );
      tnlAdaptiveRgCSRMatrix< T > argcsrMatrix( "test-matrix:TriDiagonal" );
      setBcsstk20Matrix( csrMatrix, argcsrMatrix );
      const int size = csrMatrix. getSize();

      tnlLongVector< T > x( "x" ), b1( "b1" ), b2( "b2" );
      x. setSize( size );
      b1. setSize( size );
      b2. setSize( size );
      x. setValue( 1.0 );
      csrMatrix. vectorProduct( x, b1 );
      argcsrMatrix. vectorProduct( x, b2 );

      cout << b1 << endl;
      cout << b2 << endl;

      CPPUNIT_ASSERT( b1 == b2 );
   }




};

#endif /* TNLAdaptiveRgCSRMATRIXTESTER_H_ */
