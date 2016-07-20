/***************************************************************************
                          tnlStaticArrayTester.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLSTATICARRAYTESTER_H_
#define TNLSTATICARRAYTESTER_H_

#ifdef HAVE_CPPUNIT
#include <sstream>
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <TNL/Arrays/StaticArray.h>
#include <TNL/Arrays/SharedArray.h>
#include <TNL/Arrays/ConstSharedArray.h>

using namespace TNL;
using namespace TNL::Arrays;

class testingClassForStaticArrayTester
{
   public:

      static String getType()
      {
         return String( "testingClassForStaticArrayTester" );
      };
};

String getType( const testingClassForStaticArrayTester& c )
{
   return String( "testingClassForStaticArrayTester" );
};

template< int Size, typename ElementType >
class tnlStaticArrayTester : public CppUnit :: TestCase
{
   public:

   typedef tnlStaticArrayTester< Size, ElementType > StaticArrayTester;
   typedef CppUnit :: TestCaller< StaticArrayTester > TestCaller;

   tnlStaticArrayTester(){};

   virtual
   ~tnlStaticArrayTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "ArrayTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new TestCaller( "testConstructors", &StaticArrayTester::testConstructors ) );
      suiteOfTests -> addTest( new TestCaller( "testCoordinatesGetter", &StaticArrayTester::testCoordinatesGetter ) );
      suiteOfTests -> addTest( new TestCaller( "testComparisonOperator", &StaticArrayTester::testComparisonOperator ) );
      suiteOfTests -> addTest( new TestCaller( "testAssignmentOperator", &StaticArrayTester::testAssignmentOperator ) );
      suiteOfTests -> addTest( new TestCaller( "testLoadAndSave", &StaticArrayTester::testLoadAndSave ) );
      suiteOfTests -> addTest( new TestCaller( "testSort", &StaticArrayTester::testSort ) );
      suiteOfTests -> addTest( new TestCaller( "testStreamOperator", &StaticArrayTester::testStreamOperator ) );
      suiteOfTests -> addTest( new TestCaller( "testBindToSharedArray", &StaticArrayTester::testBindToSharedArray ) );
      return suiteOfTests;
   }

   void testConstructors()
   {
      ElementType data[ Size ];
      for( int i = 0; i < Size; i++ )
         data[ i ] = i;

      tnlStaticArray< Size, ElementType > u1( data );
      for( int i = 0; i < Size; i++ )
         CPPUNIT_ASSERT( u1[ i ] == data[ i ] );

      tnlStaticArray< Size, ElementType > u2( 7 );
      for( int i = 0; i < Size; i++ )
         CPPUNIT_ASSERT( u2[ i ] == 7 );

      tnlStaticArray< Size, ElementType > u3( u1 );
      for( int i = 0; i < Size; i++ )
         CPPUNIT_ASSERT( u3[ i ] == u1[ i ] );
   }

   template< typename Element >
   void checkCoordinates( const tnlStaticArray< 1, Element >& u )
   {
      CPPUNIT_ASSERT( u.x() == 0 );
   }

   template< typename Element >
   void checkCoordinates( const tnlStaticArray< 2, Element >& u )
   {
      CPPUNIT_ASSERT( u.x() == 0 );
      CPPUNIT_ASSERT( u.y() == 1 );
   }

   template< typename Element >
   void checkCoordinates( const tnlStaticArray< 3, Element >& u )
   {
      CPPUNIT_ASSERT( u.x() == 0 );
      CPPUNIT_ASSERT( u.y() == 1 );
      CPPUNIT_ASSERT( u.z() == 2 );
   }
 
   template< int _Size, typename Element >
   void checkCoordinates( const tnlStaticArray< _Size, Element >& u )
   {
   }

   void testCoordinatesGetter()
   {
      tnlStaticArray< Size, ElementType > u;
      for( int i = 0; i < Size; i++ )
         u[ i ] = i;

      checkCoordinates( u );
   }

   void testComparisonOperator()
   {
      tnlStaticArray< Size, ElementType > u1, u2, u3;

      for( int i = 0; i < Size; i++ )
      {
         u1[ i ] = 1;
         u2[ i ] = i;
         u3[ i ] = i;
      }

      CPPUNIT_ASSERT( u1 == u1 );
      CPPUNIT_ASSERT( u1 != u2 );
      CPPUNIT_ASSERT( u2 == u3 );
   }

   void testAssignmentOperator()
   {
      tnlStaticArray< Size, ElementType > u1, u2, u3;

      for( int i = 0; i < Size; i++ )
      {
         u1[ i ] = 1;
         u2[ i ] = i;
      }

      u3 = u1;
      CPPUNIT_ASSERT( u3 == u1 );
      CPPUNIT_ASSERT( u3 != u2 );

      u3 = u2;
      CPPUNIT_ASSERT( u3 == u2 );
      CPPUNIT_ASSERT( u3 != u1 );
   }

   void testLoadAndSave()
   {
      tnlStaticArray< Size, ElementType > u1( 7 ), u2( 0 );
      File file;
      file.open( "tnl-static-array-test.tnl", tnlWriteMode );
      u1.save( file );
      file.close();
      file.open( "tnl-static-array-test.tnl", tnlReadMode );
      u2.load( file );
      file.close();

      CPPUNIT_ASSERT( u1 == u2 );
   }

   void testSort()
   {
      tnlStaticArray< Size, ElementType > u;
      for( int i = 0; i < Size; i++ )
         u[ i ] = Size - i - 1;
      u.sort();

      for( int i = 0; i < Size; i++ )
         CPPUNIT_ASSERT( u[ i ] == i );
   }

   void testStreamOperator()
   {
      tnlStaticArray< Size, ElementType > u;
      std::stringstream testStream;
      testStream << u;
   }

   void testBindToSharedArray()
   {
      tnlStaticArray< Size, ElementType > a;
      for( int i = 0; i < Size; i++ )
         a[ i ] = i+1;

      tnlSharedArray< ElementType, tnlHost > sharedArray;
      sharedArray.bind( a );
      for( int i = 0; i < Size; i++ )
         CPPUNIT_ASSERT( a[ i ] == sharedArray[ i ] );

      tnlConstSharedArray< ElementType, tnlHost > constSharedArray;
      constSharedArray.bind( a );
      for( int i = 0; i < Size; i++ )
         CPPUNIT_ASSERT( a[ i ] == constSharedArray[ i ] );

   }
};

#endif /* HAVE_CPPUNIT */


#endif /* TNLSTATICARRAYTESTER_H_ */
