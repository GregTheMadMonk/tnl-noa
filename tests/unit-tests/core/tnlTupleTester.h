/***************************************************************************
                          tnlStaticVectorTester.h  -  description
                             -------------------
    begin                : Dec 4, 2010
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

#ifndef TNLVECTORTESTER_H_
#define TNLVECTORTESTER_H_
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/tnlFile.h>
#include <core/vectors/tnlStaticVector.h>

template< int Size, typename Real > class tnlStaticVectorTester : public CppUnit :: TestCase
{
   public:
   tnlStaticVectorTester(){};

   virtual
   ~tnlStaticVectorTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlStaticVectorTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlStaticVectorTester< Size, Real > >(
                               "testConstructors",
                               & tnlStaticVectorTester< Size, Real > :: testConstructors )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlStaticVectorTester< Size, Real > >(
                               "testOperators",
                               & tnlStaticVectorTester< Size, Real > :: testOperators )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlStaticVectorTester< Size, Real > >(
                               "testSaveAndLoad",
                               & tnlStaticVectorTester< Size, Real > :: testSaveAndLoad )
                              );
      return suiteOfTests;
   };

   void testConstructors()
   {
      tnlStaticVector< Size, Real > v1;
      tnlStaticVector< Size, Real > v2( ( Real ) 1 );
      for( int i = 0; i < Size; i ++ )
      {
         CPPUNIT_ASSERT( v1[ i ] == ( Real ) 0 );
         CPPUNIT_ASSERT( v2[ i ] == ( Real ) 1 );
      }

      if( Size == 2 )
      {
         tnlStaticVector< Size, Real > v3( ( Real ) 1, ( Real ) 2 );
         CPPUNIT_ASSERT( v3[ 0 ] == ( Real ) 1 );
         CPPUNIT_ASSERT( v3[ 1 ] == ( Real ) 2 );
      }
      if( Size == 3 )
      {
         tnlStaticVector< Size, Real > v3( ( Real ) 1, ( Real ) 2, ( Real ) 3 );
         CPPUNIT_ASSERT( v3[ 0 ] == ( Real ) 1 );
         CPPUNIT_ASSERT( v3[ 1 ] == ( Real ) 2 );
         CPPUNIT_ASSERT( v3[ 2 ] == ( Real ) 3 );
      }
   };

   void testOperators()
   {
      tnlStaticVector< Size, Real > v1( ( Real ) 1 );
      tnlStaticVector< Size, Real > v2( ( Real ) 2 );
      tnlStaticVector< Size, Real > v3( ( Real ) 0 );

      v3 += v1;
      for( int i = 0; i < Size; i ++ )
         CPPUNIT_ASSERT( v3[ i ] == v1[ i ] );

      v3 -= v1;
      for( int i = 0; i < Size; i ++ )
         CPPUNIT_ASSERT( v3[ i ] == ( Real ) 0 );

      v2 *= ( Real ) 2;
      for( int i = 0; i < Size; i ++ )
         CPPUNIT_ASSERT( v2[ i ] == ( Real ) 4 );

      v3 = v1 + v2;
      for( int i = 0; i < Size; i ++ )
         CPPUNIT_ASSERT( v3[ i ] == ( Real ) 5 );

      v3 = v2 - v1;
      for( int i = 0; i < Size; i ++ )
         CPPUNIT_ASSERT( v3[ i ] == ( Real ) 3 );

      v3 = v1 * ( Real ) 2;
      for( int i = 0; i < Size; i ++ )
         CPPUNIT_ASSERT( v3[ i ] == ( Real ) 2 );

      v3 = ( ( Real ) 2 ) * v1;
      for( int i = 0; i < Size; i ++ )
         CPPUNIT_ASSERT( v3[ i ] == ( Real ) 2 );

      v3 = v1;
      for( int i = 0; i < Size; i ++ )
         CPPUNIT_ASSERT( v3[ i ] == ( Real ) 1 );

      CPPUNIT_ASSERT( v3 == v1 );
      CPPUNIT_ASSERT( v3 != v2 );
      CPPUNIT_ASSERT( v3 < v2 );
      CPPUNIT_ASSERT( v3 <= v2 );
      CPPUNIT_ASSERT( v3 <= v3 );
      CPPUNIT_ASSERT( v3 >= v3 );
      CPPUNIT_ASSERT( v2 >= v3 );
      CPPUNIT_ASSERT( v2 >= v2 );
   };

   void testSaveAndLoad()
   {
      tnlStaticVector< Size, Real > v1( ( Real ) 1 );
      tnlStaticVector< Size, Real > v2;
      tnlFile file;
      file. open( "tnlStaticVectorTest.bin", tnlWriteMode );
      v1. save( file );
      file. close();
      file. open( "tnlStaticVectorTest.bin", tnlReadMode );
      v2. load( file );
      file. close();
      CPPUNIT_ASSERT( v1 == v2 );
   };
};


#endif /* TNLVECTORTESTER_H_ */
