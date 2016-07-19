/***************************************************************************
                          tnlListTester.h  -  description
                             -------------------
    begin                : Feb 15, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLLISTTESTER_H_
#define TNLLISTTESTER_H_

#ifdef HAVE_CPPUNIT

#include <string.h>
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <TNL/core/tnlList.h>
#include <TNL/core/tnlFile.h>

using namespace TNL;

class tnlListTester : public CppUnit :: TestCase
{
   public:

   typedef tnlListTester ListTester;
   typedef CppUnit :: TestCaller< ListTester > TestCaller;

   tnlListTester(){};

   virtual
   ~tnlListTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlListTester" );
      CppUnit :: TestResult result;
      //suiteOfTests -> addTest( new TestCaller( "testBasicConstructor", &tnlStringTester :: testBasicConstructor ) );
       return suiteOfTests;
   }


};

#endif /* HAVE_CPPUNIT */


#endif /* TNLLISTTESTER_H_ */
