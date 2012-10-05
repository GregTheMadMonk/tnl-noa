/***************************************************************************
                          tnlStringTester.h -  description
                             -------------------
    begin                : Oct 4, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
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

#ifndef TNLOBJECTTESTER_H_
#define TNLOBJECTTESTER_H_

#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/tnlObject.h>
#include <core/tnlFile.h>

class tnlObjectTester : public CppUnit :: TestCase
{
   public:
   tnlObjectTester(){};

   virtual
   ~tnlObjectTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlObjectTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlObjectTester >(
                               "testObjectSave",
                               & tnlObjectTester :: testObjectSave )
                              );
       return suiteOfTests;
   }

   void testObjectSave()
   {
      tnlObject obj( "testing-object" );
      tnlFile file;
      file. open( "test-file.tnl", tnlWriteMode, tnlCompressionBzip2 );
      obj. save( file );
   };

};

#endif /* TNLOBJECTTESTER_H_ */
