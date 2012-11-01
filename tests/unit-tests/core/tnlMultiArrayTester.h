/***************************************************************************
                          tnlMultiArrayTester.h  -  description
                             -------------------
    begin                : Nov 25, 2010
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

#ifndef TNLARRAYTESTER_H_
#define TNLARRAYTESTER_H_
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/tnlMultiArray.h>
#include <core/tnlFile.h>

template< typename Real, typename device, typename Index > class tnlMultiArrayTester : public CppUnit :: TestCase
{
   public:
   tnlMultiArrayTester(){};

   virtual
   ~tnlMultiArrayTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlMultiArrayTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlMultiArrayTester< Real, device, Index > >(
                               "testConstructors",
                               & tnlMultiArrayTester< Real, device, Index > :: testConstructors )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlMultiArrayTester< Real, device, Index > >(
                               "testSetDimensions",
                               & tnlMultiArrayTester< Real, device, Index > :: testSetDimensions )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlMultiArrayTester< Real, device, Index > >(
                               "testOperators",
                               & tnlMultiArrayTester< Real, device, Index > :: testOperators )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlMultiArrayTester< Real, device, Index > >(
                               "testSaveAndLoad",
                               & tnlMultiArrayTester< Real, device, Index > :: testSaveAndLoad )
                              );

      return suiteOfTests;
   };

   void testConstructors()
   {
      tnlMultiArray< 3, Real, device, Index > array( "tnlMultiArrayTester :: array" );
      tnlMultiArray< 3, Real, device, Index > array2( "tnlMultiArrayTester :: array2", array );
      Real testData[ 1000 ];
      array2. setSharedData( testData, tnlTuple< 3, Index >( 10 ) );
      CPPUNIT_ASSERT( array2. getDimensions() == ( tnlTuple< 3, Index >( 10 ) ) );
   };

   void testSetDimensions()
   {
      tnlMultiArray< 1, Real, device, Index > u1( "tnlMultiArrayTester: u1" );
      u1. setDimensions( tnlTuple< 1, Index >( 10 ) );
      u1. setValue( ( Real ) 1 );
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( u1. getElement( i ) == ( Real ) 1 );

      for( int i = 0; i < 10; i ++ )
         u1. setElement( i, ( Real ) i );
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( u1. getElement( i ) == ( Real ) i );


      tnlMultiArray< 2, Real, device, Index > u2( "tnlMultiArrayTester: u2" );
      u2. setDimensions( tnlTuple< 2, Index >( 10 ) );
      u2. setValue( ( Real ) 1 );
      for( int i = 0; i < 10; i ++ )
         for( int j = 0; j < 10; j ++ )
               CPPUNIT_ASSERT( u2. getElement( i, j ) == ( Real ) 1 );

      for( int i = 0; i < 10; i ++ )
         for( int j = 0; j < 10; j ++ )
               u2. setElement( i, j, ( Real ) ( i + j ) );

      for( int i = 0; i < 10; i ++ )
         for( int j = 0; j < 10; j ++ )
               CPPUNIT_ASSERT( u2. getElement( i, j ) == ( Real ) ( i + j ) );

      tnlMultiArray< 3, Real, device, Index > u3( "tnlMultiArrayTester: u3" );
      u3. setDimensions( tnlTuple< 3, Index >( 10 ) );
      u3. setValue( ( Real ) 1 );
      for( int i = 0; i < 10; i ++ )
         for( int j = 0; j < 10; j ++ )
            for( int k = 0; k < 10; k ++ )
               CPPUNIT_ASSERT( u3. getElement( i, j, k ) == ( Real ) 1 );

      for( int i = 0; i < 10; i ++ )
         for( int j = 0; j < 10; j ++ )
            for( int k = 0; k < 10; k ++ )
               u3. setElement( i, j, k, ( Real ) ( i + j + k ) );

      for( int i = 0; i < 10; i ++ )
         for( int j = 0; j < 10; j ++ )
            for( int k = 0; k < 10; k ++ )
               CPPUNIT_ASSERT( u3. getElement( i, j, k ) == ( Real ) ( i + j + k ) );
   };

   void testOperators()
   {
      tnlMultiArray< 1, Real, device, Index > u1( "tnlMultiArrayTester:u1" );
      tnlMultiArray< 1, Real, device, Index > v1( "tnlMultiArrayTester:v1" );
      u1. setDimensions( tnlTuple< 1, Index >( 10 ) );
      v1. setDimensions( tnlTuple< 1, Index >( 10 ) );
      u1. setValue( ( Real ) 1 );
      v1. setValue( ( Real ) 1 );
      CPPUNIT_ASSERT( u1 == v1 );

      v1. setValue( ( Real ) 2 );
      CPPUNIT_ASSERT( ! ( u1 == v1 ) );
      CPPUNIT_ASSERT( u1 != v1 );

      v1 = u1;
      CPPUNIT_ASSERT( u1 == v1 );

      tnlMultiArray< 2, Real, device, Index > u2( "tnlMultiArrayTester:u2" );
      tnlMultiArray< 2, Real, device, Index > v2( "tnlMultiArrayTester:v2" );
      u2. setDimensions( tnlTuple< 2, Index >( 10 ) );
      v2. setDimensions( tnlTuple< 2, Index >( 10 ) );
      u2. setValue( ( Real ) 1 );
      v2. setValue( ( Real ) 1 );
      CPPUNIT_ASSERT( u2 == v2 );

      v2. setValue( ( Real ) 2 );
      CPPUNIT_ASSERT( ! ( u2 == v2 ) );
      CPPUNIT_ASSERT( u2 != v2 );

      v2 = u2;
      CPPUNIT_ASSERT( u2 == v2 );

      tnlMultiArray< 3, Real, device, Index > u3( "tnlMultiArrayTester:u3" );
      tnlMultiArray< 3, Real, device, Index > v3( "tnlMultiArrayTester:v3" );
      u3. setDimensions( tnlTuple< 3, Index >( 10 ) );
      v3. setDimensions( tnlTuple< 3, Index >( 10 ) );
      u3. setValue( ( Real ) 1 );
      v3. setValue( ( Real ) 1 );
      CPPUNIT_ASSERT( u3 == v3 );

      v3. setValue( ( Real ) 2 );
      CPPUNIT_ASSERT( ! ( u3 == v3 ) );
      CPPUNIT_ASSERT( u3 != v3 );

      v3 = u3;
      CPPUNIT_ASSERT( u3 == v3 );
   }

   void testSaveAndLoad()
   {
      tnlFile file;
      tnlMultiArray< 1, Real, device, Index > u1( "tnlMultiArrayTester:u1" );
      tnlMultiArray< 1, Real, device, Index > v1( "tnlMultiArrayTester:v1" );
      u1. setDimensions( tnlTuple< 1, Index >( 10 ) );
      u1. setValue( ( Real ) 1 );
      file. open( "tnlMultiArrayTester-file.bin", tnlWriteMode );
      u1. save( file );
      file. close();
      file. open( "tnlMultiArrayTester-file.bin", tnlReadMode );
      v1. load( file );
      file. close();
      CPPUNIT_ASSERT( u1 == v1 );

      tnlMultiArray< 2, Real, device, Index > u2( "tnlMultiArrayTester:u2" );
      tnlMultiArray< 2, Real, device, Index > v2( "tnlMultiArrayTester:v2" );
      u2. setDimensions( tnlTuple< 2, Index >( 10 ) );
      u2. setValue( ( Real ) 1 );
      file. open( "tnlMultiArrayTester-file.bin", tnlWriteMode );
      u2. save( file );
      file. close();
      file. open( "tnlMultiArrayTester-file.bin", tnlReadMode );
      v2. load( file );
      file. close();
      CPPUNIT_ASSERT( u2 == v2 );

      tnlMultiArray< 3, Real, device, Index > u3( "tnlMultiArrayTester:u3" );
      tnlMultiArray< 3, Real, device, Index > v3( "tnlMultiArrayTester:v3" );
      u3. setDimensions( tnlTuple< 3, Index >( 10 ) );
      u3. setValue( ( Real ) 1 );
      file. open( "tnlMultiArrayTester-file.bin", tnlWriteMode );
      u3. save( file );
      file. close();
      file. open( "tnlMultiArrayTester-file.bin", tnlReadMode );
      v3. load( file );
      file. close();
      CPPUNIT_ASSERT( u3 == v3 );
   }

};
#endif /* TNLARRAYTESTER_H_ */
