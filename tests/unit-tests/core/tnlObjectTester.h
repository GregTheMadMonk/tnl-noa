/***************************************************************************
                          tnlObjectTester.h -  description
                             -------------------
    begin                : Oct 4, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <TNL/Object.h>
#include <TNL/File.h>

using namespace TNL;

class ObjectTester : public CppUnit :: TestCase
{
   public:
   ObjectTester(){};

   virtual
   ~ObjectTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "ObjectTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< ObjectTester >(
                               "testObjectSave",
                               & ObjectTester :: testObjectSave )
                              );
       return suiteOfTests;
   }

   void testObjectSave()
   {
      Object obj;
      File file;
      file. open( "test-file.tnl", tnlWriteMode );
      obj.save( file );
   };

};

#else /* HAVE_CPPUNIT */
class ObjectTester{};
#endif  /* HAVE_CPPUNIT */

