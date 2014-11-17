/***************************************************************************
                          tnlPDEOperatorEocTester.h  -  description
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

#ifndef TNLPDEOPERATOREOCTESTER_H_
#define TNLPDEOPERATOREOCTESTER_H_

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <unit-tests/tnlApproximationError.h>
#include <unit-tests/operators/tnlPDEOperatorEocTestSetter.h>
#include <unit-tests/operators/tnlPDEOperatorEocTestResult.h>

template< typename ApproximateOperator,
          typename ExactOperator,
          typename TestFunction,
          typename ApproximationMethod,
          int MeshSize,
          bool verbose = false >
class tnlPDEOperatorEocTester : public CppUnit :: TestCase
{
   public:
   typedef tnlPDEOperatorEocTester< ApproximateOperator, ExactOperator, TestFunction, ApproximationMethod, MeshSize, verbose > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;
   typedef typename ApproximateOperator::MeshType MeshType;
   typedef typename ApproximateOperator::RealType RealType;
   typedef typename ApproximateOperator::IndexType IndexType;
   typedef tnlPDEOperatorEocTestSetter< ApproximateOperator, ExactOperator, ApproximationMethod, MeshType, TestFunction > TestSetter;
   typedef tnlPDEOperatorEocTestResult< ApproximateOperator, ApproximationMethod, TestFunction > TestResult;

   tnlPDEOperatorEocTester(){};

   virtual
   ~tnlPDEOperatorEocTester(){};

   static CppUnit :: Test* suite()
   {
      tnlString testName = tnlString( "tnlPDEOperatorEocTester< " ) +
                           ApproximateOperator::getType() + ", " +
                           ExactOperator::getType() + ", " +
                           TestFunction::getType() + ", " +
                           ApproximationMethod::getType() + ", " +
                           tnlString( MeshSize ) + ", " +
                           tnlString( verbose ) + " >";
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( testName.getString() );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "approximationTest", &TesterType::approximationTest ) );

      return suiteOfTests;
   }

   void approximationTest()
   {
      TestFunction testFunction;
      TestSetter::setFunction( testFunction );

      MeshType mesh;

      ExactOperator exactOperator;
      ApproximateOperator approximateOperator;

      TestSetter::setMesh( mesh, MeshSize );
      RealType coarseL1Err, coarseL2Err, coarseMaxErr;
      tnlApproximationError< MeshType,
                             ExactOperator,
                             ApproximateOperator,
                             TestFunction,
                             ApproximationMethod >
         ::getError( mesh,
                     exactOperator,
                     approximateOperator,
                     testFunction,
                     coarseL1Err,
                     coarseL2Err,
                     coarseMaxErr );

      TestSetter::setMesh( mesh, 2*MeshSize );
      RealType fineL1Err, fineL2Err, fineMaxErr;
      tnlApproximationError< MeshType,
                             ExactOperator,
                             ApproximateOperator,
                             TestFunction,
                             ApproximationMethod >
         ::getError( mesh,
                     exactOperator,
                     approximateOperator,
                     testFunction,
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
   }
};

#endif /* HAVE_CPPUNIT */
#endif /* TNLPDEOPERATOREOCTESTER_H_ */
