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
      /*
      TestFunction testFunction;
      TestSetter::setFunction( testFunction );

      MeshType mesh;

      ExactOperator exactOperator;
      ApproximateOperator approximateOperator;

      TestSetter::setMesh( mesh, MeshSize );
      RealType coarseL1Err, coarseL2Err, coarseMaxErr;
      tnlApproximationError< ExactOperator,
                             ApproximateOperator,
                             MeshEntityType,
                             TestFunction,
                             writeFunctions >
         ::getError( exactOperator,
                     approximateOperator,
                     testFunction,
                     mesh,
                     coarseL1Err,
                     coarseL2Err,
                     coarseMaxErr );

      TestSetter::setMesh( mesh, 2*MeshSize );
      RealType fineL1Err, fineL2Err, fineMaxErr;
      tnlApproximationError< ExactOperator,
                             ApproximateOperator,
                             MeshEntityType,
                             TestFunction,
                             writeFunctions >
         ::getError( exactOperator,
                     approximateOperator,
                     testFunction,
                     mesh,
                     fineL1Err,
                     fineL2Err,
                     fineMaxErr );
      RealType l1Eoc = log( coarseL1Err / fineL1Err) / log( 2.0 );
      RealType l2Eoc = log( coarseL2Err / fineL2Err) / log( 2.0 );
      RealType maxEoc = log( coarseMaxErr / fineMaxErr) / log( 2.0 );

      if( verbose )
      {
         cerr << "Coarse mesh: L1Err = " << coarseL1Err << " L2Err = " << coarseL2Err << " maxErr = " << coarseMaxErr << endl;
         cerr << "Fine mesh: L1Err = " << fineL1Err << " L2Err = " << fineL2Err << " maxErr = " << fineMaxErr << endl;
         cerr << "L1Eoc = " << l1Eoc << " L2Eoc = " << l2Eoc << " maxEoc = " << maxEoc << endl;
      }

      CPPUNIT_ASSERT( fabs( l1Eoc - TestResult::getL1Eoc() ) < TestResult::getL1Tolerance() );
      CPPUNIT_ASSERT( fabs( l2Eoc - TestResult::getL2Eoc() ) < TestResult::getL2Tolerance() );
      CPPUNIT_ASSERT( fabs( maxEoc - TestResult::getMaxEoc() ) < TestResult::getMaxTolerance() );
      */
   }
};

#endif /* HAVE_CPPUNIT */
#endif /* TNLPDEOPERATOREOCUNITTEST_H_ */
