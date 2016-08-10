/***************************************************************************
                          GridOldTester.h  -  description
                             -------------------
    begin                : Dec 13, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef GridOldTESTER_H_
#define GridOldTESTER_H_
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <TNL/legacy/mesh/GridOld.h>
#include <TNL/File.h>

template< typename Real, typename device, typename Index > class GridOldTester : public CppUnit :: TestCase
{
   public:
   GridOldTester(){};

   virtual
   ~GridOldTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "GridOldTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< GridOldTester< Real, device, Index > >(
                               "testConstructors",
                               & GridOldTester< Real, device, Index > :: testConstructors )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< GridOldTester< Real, device, Index > >(
                               "testSetDomain",
                               & GridOldTester< Real, device, Index > :: testSetDomain )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< GridOldTester< Real, device, Index > >(
                               "testSaveAndLoad",
                               & GridOldTester< Real, device, Index > :: testSaveAndLoad )
                              );

      return suiteOfTests;
   };

   void testConstructors()
   {
      GridOld< 3, Real, device, Index > Grid( "GridOldTester :: Grid" );
      GridOld< 3, Real, device, Index > Grid2( "GridOldTester :: Grid2", Grid );
   };

   void testSetDomain()
   {
      GridOld< 1, Real, device, Index > u1( "GridOldTester: u1" );
      u1. setDimensions( StaticVector< 1, Index >( 11 ) );
      u1. setValue( ( Real ) 1 );
      u1. setDomain( StaticVector< 1, Real >( 0.0 ),
                     StaticVector< 1, Real >( 1.0 ) );
      CPPUNIT_ASSERT( u1. getSpaceSteps() == ( StaticVector< 1, Real >( 0.1 ) ) );
   };


   void testSaveAndLoad()
   {
      /*File file;
      GridOld< 1, Real, device, Index > u1( "GridOldTester:u1" );
      GridOld< 1, Real, device, Index > v1( "GridOldTester:v1" );
      u1. setDimensions( StaticVector< 1, Index >( 10 ) );
      u1. setValue( ( Real ) 1 );
      file. open( "GridOldTester-file.bin", tnlWriteMode );
      u1. save( file );
      file. close();
      file. open( "GridOldTester-file.bin", tnlReadMode );
      v1. load( file );
      file. close();
      CPPUNIT_ASSERT( u1 == v1 );

      GridOld< 2, Real, device, Index > u2( "GridOldTester:u2" );
      GridOld< 2, Real, device, Index > v2( "GridOldTester:v2" );
      u2. setDimensions( StaticVector< 2, Index >( 10 ) );
      u2. setValue( ( Real ) 1 );
      file. open( "GridOldTester-file.bin", tnlWriteMode );
      u2. save( file );
      file. close();
      file. open( "GridOldTester-file.bin", tnlReadMode );
      v2. load( file );
      file. close();
      CPPUNIT_ASSERT( u2 == v2 );

      GridOld< 3, Real, device, Index > u3( "GridOldTester:u3" );
      GridOld< 3, Real, device, Index > v3( "GridOldTester:v3" );
      u3. setDimensions( StaticVector< 3, Index >( 10 ) );
      u3. setValue( ( Real ) 1 );
      file. open( "GridOldTester-file.bin", tnlWriteMode );
      u3. save( file );
      file. close();
      file. open( "GridOldTester-file.bin", tnlReadMode );
      v3. load( file );
      file. close();
      CPPUNIT_ASSERT( u3 == v3 );*/
   }

};
#endif /* GridOldTESTER_H_ */
