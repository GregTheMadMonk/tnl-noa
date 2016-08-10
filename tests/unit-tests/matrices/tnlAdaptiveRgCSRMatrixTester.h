/***************************************************************************
                          tnlAdaptiveRgCSRTester.h  -  description
                             -------------------
    begin                : Jul 20, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLAdaptiveRgCSRMATRIXTESTER_H_
#define TNLAdaptiveRgCSRMATRIXTESTER_H_

#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <TNL/Matrices/CSR.h>
#include <TNL/legacy/matrices/tnlAdaptiveRgCSR.h>
#include "MatrixTester.h"

using namespace TNL;

template< class Real, typename Device > class tnlAdaptiveRgCSRTester : public CppUnit :: TestCase,
                                                                           public MatrixTester< Real >
{
   public:
   tnlAdaptiveRgCSRTester(){};

   virtual
   ~tnlAdaptiveRgCSRTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlAdaptiveRgCSRTester" );
      CppUnit :: TestResult result;

      if( Device :: getDevice() == Devices::HostDevice )
      {
         suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRTester< Real, Device > >(
                                  "ifEmptyMatrixIsStoredProperly",
                                  & tnlAdaptiveRgCSRTester< Real, Device > :: ifEmptyMatrixIsStoredProperly )
                                );
         suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRTester< Real, Device > >(
                                  "ifDiagonalMatrixIsStoredProperly",
                                  & tnlAdaptiveRgCSRTester< Real, Device > :: ifDiagonalMatrixIsStoredProperly )
                                );
         suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRTester< Real, Device > >(
                                  "ifTriDiagonalMatrixIsStoredProperly",
                                  & tnlAdaptiveRgCSRTester< Real, Device > :: ifTriDiagonalMatrixIsStoredProperly )
                                );
         suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRTester< Real, Device > >(
                                  "ifUpperTriangularMatrixIsStoredProperly",
                                  & tnlAdaptiveRgCSRTester< Real, Device > :: ifUpperTriangularMatrixIsStoredProperly )
                                );
         suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRTester< Real, Device > >(
                                  "ifFullMatrixIsStoredProperly",
                                  & tnlAdaptiveRgCSRTester< Real, Device > :: ifFullMatrixIsStoredProperly )
                                );
         suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRTester< Real, Device > >(
                                  "ifBcsstk20MatrixIsStoredProperly",
                                  & tnlAdaptiveRgCSRTester< Real, Device > :: ifBcsstk20MatrixIsStoredProperly )
                                );
      }

      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRTester< Real, Device > >(
                               "ifSpmvWithEmptyMatrixWorks",
                               & tnlAdaptiveRgCSRTester< Real, Device > :: ifSpmvWithEmptyMatrixWorks )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRTester< Real, Device > >(
                               "ifSpmvWithDiagonalMatrixWorks",
                               & tnlAdaptiveRgCSRTester< Real, Device > :: ifSpmvWithDiagonalMatrixWorks )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRTester< Real, Device > >(
                               "ifSpmvWithTriDiagonalMatrixWorks",
                               & tnlAdaptiveRgCSRTester< Real, Device > :: ifSpmvWithTriDiagonalMatrixWorks )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRTester< Real, Device > >(
                               "ifSpmvWithUpperTriangularMatrixWorks",
                               & tnlAdaptiveRgCSRTester< Real, Device > :: ifSpmvWithUpperTriangularMatrixWorks  )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRTester< Real, Device > >(
                               "ifSpmvWithFullMatrixWorks",
                               & tnlAdaptiveRgCSRTester< Real, Device > :: ifSpmvWithFullMatrixWorks  )
                             );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlAdaptiveRgCSRTester< Real, Device > >(
                               "ifSpmvWithBcsstk20MatrixWorks",
                               & tnlAdaptiveRgCSRTester< Real, Device > :: ifSpmvWithBcsstk20MatrixWorks )
                             );
      return suiteOfTests;
   }

   void ifEmptyMatrixIsStoredProperly()
   {
      const int size = 12;
      CSR< Real > csrMatrix;
      //( "test-matrix:Empty" );
      tnlAdaptiveRgCSR< Real, Device > argcsrMatrix( "test-matrix:Empty" );
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
      CSR< Real > csrMatrix;
      //( "test-matrix:Diagonal" );
      tnlAdaptiveRgCSR< Real, Device > argcsrMatrix( "test-matrix:Diagonal" );
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
      CSR< Real > csrMatrix;
      //( "test-matrix:Tridiagonal" );
      tnlAdaptiveRgCSR< Real, Device > argcsrMatrix( "test-matrix:Tridiagonal" );
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
      CSR< Real > csrMatrix;
      //( "test-matrix:upperTriangular" );
      tnlAdaptiveRgCSR< Real, Device > argcsrMatrix( "test-matrix:upperTriangular" );
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
      CSR< Real > csrMatrix;
      //( "test-matrix:full" );
      tnlAdaptiveRgCSR< Real, Device > argcsrMatrix( "test-matrix:full" );
      const int size = 12;
      this->setFullMatrix( csrMatrix, size );
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
      CSR< Real > csrMatrix;
      //( "test-matrix:bcsstk20" );
      tnlAdaptiveRgCSR< Real, Device > argcsrMatrix( "test-matrix:bcsstk20" );
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
      CSR< Real > csrMatrix;//( "test-matrix:Empty" );
      tnlAdaptiveRgCSR< Real, Device > argcsrMatrix( "test-matrix:Empty" );
      this->setEmptyMatrix( csrMatrix, size );
      if( Device :: getDevice() == Devices::HostDevice )
         argcsrMatrix. copyFrom( csrMatrix );
      else
      {
         tnlAdaptiveRgCSR< Real, Devices::Host > hostArgcsrMatrix( "test-matrix:host-aux" );
         hostArgcsrMatrix. copyFrom( csrMatrix );
         argcsrMatrix. copyFrom( hostArgcsrMatrix );
      }

      Vector< Real, Devices::Host > x1( "x1" ), b1( "b1" );
      Vector< Real, Device > x2( "x2" ), b2( "b2" );
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
      CSR< Real > csrMatrix;//( "test-matrix:Diagonal" );
      tnlAdaptiveRgCSR< Real, Device > argcsrMatrix( "test-matrix:Diagonal" );
      this->setDiagonalMatrix( csrMatrix, size );
      if( Device :: getDevice() == Devices::HostDevice )
         argcsrMatrix. copyFrom( csrMatrix );
      else
      {
         tnlAdaptiveRgCSR< Real, Devices::Host > hostArgcsrMatrix( "test-matrix:host-aux" );
         hostArgcsrMatrix. copyFrom( csrMatrix );
         argcsrMatrix. copyFrom( hostArgcsrMatrix );
      }

      Vector< Real, Devices::Host > x1( "x1" ), b1( "b1" );
      Vector< Real, Device > x2( "x2" ), b2( "b2" );
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
      CSR< Real > csrMatrix;//( "test-matrix:TriDiagonal" );
      tnlAdaptiveRgCSR< Real, Device > argcsrMatrix( "test-matrix:TriDiagonal" );
      this->setTridiagonal( csrMatrix, size );
      if( Device :: getDevice() == Devices::HostDevice )
         argcsrMatrix. copyFrom( csrMatrix );
      else
      {
         tnlAdaptiveRgCSR< Real, Devices::Host > hostArgcsrMatrix( "test-matrix:host-aux" );
         hostArgcsrMatrix. copyFrom( csrMatrix );
         argcsrMatrix. copyFrom( hostArgcsrMatrix );
      }

      Vector< Real, Devices::Host > x1( "x1" ), b1( "b1" );
      Vector< Real, Device > x2( "x2" ), b2( "b2" );
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
      CSR< Real > csrMatrix;//( "test-matrix:TriDiagonal" );
      tnlAdaptiveRgCSR< Real, Device > argcsrMatrix( "test-matrix:TriDiagonal" );
      this->setUpperTriangularMatrix( csrMatrix, size );
      if( Device :: getDevice() == Devices::HostDevice )
         argcsrMatrix. copyFrom( csrMatrix );
      else
      {
         tnlAdaptiveRgCSR< Real, Devices::Host > hostArgcsrMatrix( "test-matrix:host-aux" );
         hostArgcsrMatrix. copyFrom( csrMatrix );
         argcsrMatrix. copyFrom( hostArgcsrMatrix );
      }

      Vector< Real, Devices::Host > x1( "x1" ), b1( "b1" );
      Vector< Real, Device > x2( "x2" ), b2( "b2" );
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
      CSR< Real > csrMatrix;//( "test-matrix:full" );
      tnlAdaptiveRgCSR< Real, Device > argcsrMatrix( "test-matrix:full" );
      this->setFullMatrix( csrMatrix, size );
      if( Device :: getDevice() == Devices::HostDevice )
         argcsrMatrix. copyFrom( csrMatrix );
      else
      {
         tnlAdaptiveRgCSR< Real, Devices::Host > hostArgcsrMatrix( "test-matrix:host-aux" );
         hostArgcsrMatrix. copyFrom( csrMatrix );
         argcsrMatrix. copyFrom( hostArgcsrMatrix );
      }

      Vector< Real, Devices::Host > x1( "x1" ), b1( "b1" );
      Vector< Real, Device > x2( "x2" ), b2( "b2" );
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
      CSR< Real > csrMatrix;//( "test-matrix:TriDiagonal" );
      tnlAdaptiveRgCSR< Real, Device > argcsrMatrix( "test-matrix:TriDiagonal" );
      this->setBcsstk20Matrix( csrMatrix );
      if( Device :: getDevice() == Devices::HostDevice )
         argcsrMatrix. copyFrom( csrMatrix );
      else
      {
         tnlAdaptiveRgCSR< Real, Devices::Host > hostArgcsrMatrix( "test-matrix:host-aux" );
         hostArgcsrMatrix. copyFrom( csrMatrix );
         argcsrMatrix. copyFrom( hostArgcsrMatrix );
      }

      const int size = csrMatrix. getRows();
      Vector< Real, Devices::Host > x1( "x1" ), b1( "b1" );
      Vector< Real, Device > x2( "x2" ), b2( "b2" );
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
         //f << refB[ j ] << " - " << host_b[ j ] << " = "  << refB[ j ] - host_b[ j ] <<  std::endl;
         Real error( 0.0 );
         if( b1. getElement( j ) != 0.0 )
            error = ( Real ) fabs( b1. getElement( j ) - b2. getElement( j ) ) /  ( Real ) fabs( b1. getElement( j ) );
         else
            error = ( Real ) fabs( b2. getElement( j ) );
         maxError = max( maxError, error );
      }
      CPPUNIT_ASSERT( maxError < 1.0e-12 );
   }
};

#endif /* TNLAdaptiveRgCSRMATRIXTESTER_H_ */
