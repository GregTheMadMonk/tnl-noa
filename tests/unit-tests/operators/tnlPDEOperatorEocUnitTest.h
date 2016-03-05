/***************************************************************************
                          tnlPDEOperatorEocUnitTest.h  -  description
                             -------------------
    begin                : Aug 30, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLPDEOPERATOREOCUNITTEST_H_
#define TNLPDEOPERATOREOCUNITTEST_H_

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>

template< typename OperatorTest >
class tnlPDEOperatorEocUnitTest : public CppUnit :: TestCase
{
   public:
   typedef tnlPDEOperatorEocUnitTest< OperatorTest > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;

   tnlPDEOperatorEocUnitTest(){};

   virtual
   ~tnlPDEOperatorEocUnitTest(){};

   static CppUnit :: Test* suite()
   {
      tnlString testName = OperatorTest::getType();
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( testName.getString() );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new TestCallerType( testName.getString(), &TesterType::approximationTest ) );

      return suiteOfTests;
   }

   void approximationTest()
   {
      OperatorTest operatorTest;
      operatorTest.setupTest();
      operatorTest.runUnitTest();
   }
};

#endif /* HAVE_CPPUNIT */
#endif /* TNLPDEOPERATOREOCUNITTEST_H_ */
