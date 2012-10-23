/***************************************************************************
                          tnlArrayTester.h  -  description
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
#include <core/tnlArray.h>
#include <core/tnlFile.h>

template< typename Real, typename device, typename Index > class tnlArrayTester : public CppUnit :: TestCase
{
   public:
   tnlArrayTester(){};

   virtual
   ~tnlArrayTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlArrayTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayTester< Real, device, Index > >(
                               "testConstructors",
                               & tnlArrayTester< Real, device, Index > :: testConstructors )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayTester< Real, device, Index > >(
                               "testSetDimensions",
                               & tnlArrayTester< Real, device, Index > :: testSetDimensions )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayTester< Real, device, Index > >(
                               "testOperators",
                               & tnlArrayTester< Real, device, Index > :: testOperators )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlArrayTester< Real, device, Index > >(
                               "testSaveAndLoad",
                               & tnlArrayTester< Real, device, Index > :: testSaveAndLoad )
                              );

      return suiteOfTests;
   };

   void testConstructors()
   {
      tnlArray< 3, Real, device, Index > array( "tnlArrayTester :: array" );
      tnlArray< 3, Real, device, Index > array2( "tnlArrayTester :: array2", array );
      Real testData[ 1000 ];
      array2. setSharedData( testData, tnlTuple< 3, Index >( 10 ) );
      CPPUNIT_ASSERT( array2. getDimensions() == ( tnlTuple< 3, Index >( 10 ) ) );
   };

   void testSetDimensions()
   {
      tnlArray< 1, Real, device, Index > u1( "tnlArrayTester: u1" );
      u1. setDimensions( tnlTuple< 1, Index >( 10 ) );
      u1. setValue( ( Real ) 1 );
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( u1. getElement( i ) == ( Real ) 1 );

      for( int i = 0; i < 10; i ++ )
         u1. setElement( i, ( Real ) i );
      for( int i = 0; i < 10; i ++ )
         CPPUNIT_ASSERT( u1. getElement( i ) == ( Real ) i );


      tnlArray< 2, Real, device, Index > u2( "tnlArrayTester: u2" );
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

      tnlArray< 3, Real, device, Index > u3( "tnlArrayTester: u3" );
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
      tnlArray< 1, Real, device, Index > u1( "tnlArrayTester:u1" );
      tnlArray< 1, Real, device, Index > v1( "tnlArrayTester:v1" );
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

      tnlArray< 2, Real, device, Index > u2( "tnlArrayTester:u2" );
      tnlArray< 2, Real, device, Index > v2( "tnlArrayTester:v2" );
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

      tnlArray< 3, Real, device, Index > u3( "tnlArrayTester:u3" );
      tnlArray< 3, Real, device, Index > v3( "tnlArrayTester:v3" );
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
      tnlArray< 1, Real, device, Index > u1( "tnlArrayTester:u1" );
      tnlArray< 1, Real, device, Index > v1( "tnlArrayTester:v1" );
      u1. setDimensions( tnlTuple< 1, Index >( 10 ) );
      u1. setValue( ( Real ) 1 );
      file. open( "tnlArrayTester-file.bin", tnlWriteMode );
      u1. save( file );
      file. close();
      file. open( "tnlArrayTester-file.bin", tnlReadMode );
      v1. load( file );
      file. close();
      CPPUNIT_ASSERT( u1 == v1 );

      tnlArray< 2, Real, device, Index > u2( "tnlArrayTester:u2" );
      tnlArray< 2, Real, device, Index > v2( "tnlArrayTester:v2" );
      u2. setDimensions( tnlTuple< 2, Index >( 10 ) );
      u2. setValue( ( Real ) 1 );
      file. open( "tnlArrayTester-file.bin", tnlWriteMode );
      u2. save( file );
      file. close();
      file. open( "tnlArrayTester-file.bin", tnlReadMode );
      v2. load( file );
      file. close();
      CPPUNIT_ASSERT( u2 == v2 );

      tnlArray< 3, Real, device, Index > u3( "tnlArrayTester:u3" );
      tnlArray< 3, Real, device, Index > v3( "tnlArrayTester:v3" );
      u3. setDimensions( tnlTuple< 3, Index >( 10 ) );
      u3. setValue( ( Real ) 1 );
      file. open( "tnlArrayTester-file.bin", tnlWriteMode );
      u3. save( file );
      file. close();
      file. open( "tnlArrayTester-file.bin", tnlReadMode );
      v3. load( file );
      file. close();
      CPPUNIT_ASSERT( u3 == v3 );
   }

};
#endif /* TNLARRAYTESTER_H_ */
