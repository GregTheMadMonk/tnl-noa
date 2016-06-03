/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

/***************************************************************************
                          tnlUniquePointerTester.h -  description
                             -------------------
    begin                : May 28, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#pragma once

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/tnlUniquePointer.h>
#include <core/arrays/tnlStaticArray.h>


class tnlUniquePointerTester : public CppUnit :: TestCase
{
   public:
      typedef tnlUniquePointerTester TesterType;
      typedef typename CppUnit::TestCaller< TesterType > TestCallerType;
   
   public:
   tnlUniquePointerTester(){};

   virtual
   ~tnlUniquePointerTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlUniquePointerTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "testConstructor", &TesterType::testConstructor ) );
      return suiteOfTests;
   }

   void testConstructor()
   {
      typedef tnlStaticArray< 2, int  > TestType;
      
      tnlUniquePointer< TestType, tnlHost > ptr1;
      
      ptr1->x() = 0;
      ptr1->y() = 0;
      
      CPPUNIT_ASSERT( ptr1->x() == 0 && ptr1->y() == 0 );
      
      tnlUniquePointer< TestType, tnlHost > ptr2( 1, 2 );
      
      CPPUNIT_ASSERT( ptr2->x() == 1 && ptr2->y() == 2 );
      
      ptr1 = ptr2;
      
      CPPUNIT_ASSERT( ptr1->x() == 1 && ptr1->y() == 2 );
      
   };

};

#else /* HAVE_CPPUNIT */
class tnlObjectTester{};
#endif  /* HAVE_CPPUNIT */

