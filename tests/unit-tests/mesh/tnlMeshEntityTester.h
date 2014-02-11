/***************************************************************************
                          tnlMeshEntityTester.h  -  description
                             -------------------
    begin                : Feb 11, 2014
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

#ifndef TNLMESHENTITYTESTER_H_
#define TNLMESHENTITYTESTER_H_

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <mesh/tnlMeshEntity.h>
#include <mesh/config/tnlMeshConfigBase.h>

template< typename RealType, typename Device, typename IndexType >
class tnlMeshEntityTester : public CppUnit :: TestCase
{
   public:

   typedef tnlMeshEntityTester< RealType, Device, IndexType > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;

   tnlMeshEntityTester(){};

   virtual
   ~tnlMeshEntityTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlMeshEntityTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "vertexMeshEntityTest", &TesterType::vertexMeshEntityTest ) );

      return suiteOfTests;
   }

   void vertexMeshEntityTest()
   {
      tnlMeshEntity< tnlMeshConfigBase > meshEntity;
   }

};

#endif



#endif /* TNLMESHENTITYTESTER_H_ */
