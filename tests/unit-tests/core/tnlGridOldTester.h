/***************************************************************************
                          tnlGridOldTester.h  -  description
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

#ifndef tnlGridOldTESTER_H_
#define tnlGridOldTESTER_H_
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <legacy/mesh/tnlGridOld.h>
#include <core/tnlFile.h>

template< typename Real, typename device, typename Index > class tnlGridOldTester : public CppUnit :: TestCase
{
   public:
   tnlGridOldTester(){};

   virtual
   ~tnlGridOldTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlGridOldTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlGridOldTester< Real, device, Index > >(
                               "testConstructors",
                               & tnlGridOldTester< Real, device, Index > :: testConstructors )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlGridOldTester< Real, device, Index > >(
                               "testSetDomain",
                               & tnlGridOldTester< Real, device, Index > :: testSetDomain )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlGridOldTester< Real, device, Index > >(
                               "testSaveAndLoad",
                               & tnlGridOldTester< Real, device, Index > :: testSaveAndLoad )
                              );

      return suiteOfTests;
   };

   void testConstructors()
   {
      tnlGridOld< 3, Real, device, Index > Grid( "tnlGridOldTester :: Grid" );
      tnlGridOld< 3, Real, device, Index > Grid2( "tnlGridOldTester :: Grid2", Grid );
   };

   void testSetDomain()
   {
      tnlGridOld< 1, Real, device, Index > u1( "tnlGridOldTester: u1" );
      u1. setDimensions( tnlStaticVector< 1, Index >( 11 ) );
      u1. setValue( ( Real ) 1 );
      u1. setDomain( tnlStaticVector< 1, Real >( 0.0 ),
                     tnlStaticVector< 1, Real >( 1.0 ) );
      CPPUNIT_ASSERT( u1. getSpaceSteps() == ( tnlStaticVector< 1, Real >( 0.1 ) ) );
   };


   void testSaveAndLoad()
   {
      /*tnlFile file;
      tnlGridOld< 1, Real, device, Index > u1( "tnlGridOldTester:u1" );
      tnlGridOld< 1, Real, device, Index > v1( "tnlGridOldTester:v1" );
      u1. setDimensions( tnlStaticVector< 1, Index >( 10 ) );
      u1. setValue( ( Real ) 1 );
      file. open( "tnlGridOldTester-file.bin", tnlWriteMode );
      u1. save( file );
      file. close();
      file. open( "tnlGridOldTester-file.bin", tnlReadMode );
      v1. load( file );
      file. close();
      CPPUNIT_ASSERT( u1 == v1 );

      tnlGridOld< 2, Real, device, Index > u2( "tnlGridOldTester:u2" );
      tnlGridOld< 2, Real, device, Index > v2( "tnlGridOldTester:v2" );
      u2. setDimensions( tnlStaticVector< 2, Index >( 10 ) );
      u2. setValue( ( Real ) 1 );
      file. open( "tnlGridOldTester-file.bin", tnlWriteMode );
      u2. save( file );
      file. close();
      file. open( "tnlGridOldTester-file.bin", tnlReadMode );
      v2. load( file );
      file. close();
      CPPUNIT_ASSERT( u2 == v2 );

      tnlGridOld< 3, Real, device, Index > u3( "tnlGridOldTester:u3" );
      tnlGridOld< 3, Real, device, Index > v3( "tnlGridOldTester:v3" );
      u3. setDimensions( tnlStaticVector< 3, Index >( 10 ) );
      u3. setValue( ( Real ) 1 );
      file. open( "tnlGridOldTester-file.bin", tnlWriteMode );
      u3. save( file );
      file. close();
      file. open( "tnlGridOldTester-file.bin", tnlReadMode );
      v3. load( file );
      file. close();
      CPPUNIT_ASSERT( u3 == v3 );*/
   }

};
#endif /* tnlGridOldTESTER_H_ */
