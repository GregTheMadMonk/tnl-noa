/***************************************************************************
                          tnlGridTester.h  -  description
                             -------------------
    begin                : Dec 13, 2010
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

#ifndef TNLGRIDTESTER_H_
#define TNLGRIDTESTER_H_
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <mesh/tnlGrid.h>
#include <core/tnlFile.h>

template< typename Real, tnlDevice device, typename Index > class tnlGridTester : public CppUnit :: TestCase
{
   public:
   tnlGridTester(){};

   virtual
   ~tnlGridTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlGridTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlGridTester< Real, device, Index > >(
                               "testConstructors",
                               & tnlGridTester< Real, device, Index > :: testConstructors )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlGridTester< Real, device, Index > >(
                               "testSetDomain",
                               & tnlGridTester< Real, device, Index > :: testSetDomain )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlGridTester< Real, device, Index > >(
                               "testSaveAndLoad",
                               & tnlGridTester< Real, device, Index > :: testSaveAndLoad )
                              );

      return suiteOfTests;
   };

   void testConstructors()
   {
      tnlGrid< 3, Real, device, Index > Grid( "tnlGridTester :: Grid" );
      tnlGrid< 3, Real, device, Index > Grid2( "tnlGridTester :: Grid2", Grid );
   };

   void testSetDomain()
   {
      tnlGrid< 1, Real, device, Index > u1( "tnlGridTester: u1" );
      u1. setDimensions( tnlVector< 1, Index >( 11 ) );
      u1. setValue( ( Real ) 1 );
      u1. setDomain( tnlVector< 1, Real >( 0.0 ),
                     tnlVector< 1, Real >( 1.0 ) );
      CPPUNIT_ASSERT( u1. getSpaceSteps() == ( tnlVector< 1, Real >( 0.1 ) ) );
   };


   void testSaveAndLoad()
   {
      /*tnlFile file;
      tnlGrid< 1, Real, device, Index > u1( "tnlGridTester:u1" );
      tnlGrid< 1, Real, device, Index > v1( "tnlGridTester:v1" );
      u1. setDimensions( tnlVector< 1, Index >( 10 ) );
      u1. setValue( ( Real ) 1 );
      file. open( "tnlGridTester-file.bin", tnlWriteMode );
      u1. save( file );
      file. close();
      file. open( "tnlGridTester-file.bin", tnlReadMode );
      v1. load( file );
      file. close();
      CPPUNIT_ASSERT( u1 == v1 );

      tnlGrid< 2, Real, device, Index > u2( "tnlGridTester:u2" );
      tnlGrid< 2, Real, device, Index > v2( "tnlGridTester:v2" );
      u2. setDimensions( tnlVector< 2, Index >( 10 ) );
      u2. setValue( ( Real ) 1 );
      file. open( "tnlGridTester-file.bin", tnlWriteMode );
      u2. save( file );
      file. close();
      file. open( "tnlGridTester-file.bin", tnlReadMode );
      v2. load( file );
      file. close();
      CPPUNIT_ASSERT( u2 == v2 );

      tnlGrid< 3, Real, device, Index > u3( "tnlGridTester:u3" );
      tnlGrid< 3, Real, device, Index > v3( "tnlGridTester:v3" );
      u3. setDimensions( tnlVector< 3, Index >( 10 ) );
      u3. setValue( ( Real ) 1 );
      file. open( "tnlGridTester-file.bin", tnlWriteMode );
      u3. save( file );
      file. close();
      file. open( "tnlGridTester-file.bin", tnlReadMode );
      v3. load( file );
      file. close();
      CPPUNIT_ASSERT( u3 == v3 );*/
   }

};
#endif /* TNLGRIDTESTER_H_ */
