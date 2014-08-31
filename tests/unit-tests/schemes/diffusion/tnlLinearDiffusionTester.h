/***************************************************************************
                          tnlLinearDiffusionTester.h  -  description
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

#ifndef TNLLINEARDIFFUSIONTESTER_H_
#define TNLLINEARDIFFUSIONTESTER_H_

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <schemes/diffusion/tnlLinearDiffusion.h>
#include <functions/tnlExpBumpFunction.h>
#include <tnlApproximationError.h>

template< typename MeshType,
          typename TestFunction >
class tnlLinearDiffusionTester{}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename TestFunction >
class tnlLinearDiffusionTester< tnlGrid< Dimensions, Real, Device, Index >,
                                TestFunction > : public CppUnit :: TestCase
{
   public:
   typedef tnlGrid< Dimensions, Real, Device, Index > MeshType;
   typedef tnlLinearDiffusionTester< MeshType > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;
   typedef tnlLinearDiffusionTestSetter< MeshType, FunctionType > TestSetter;
   typedef tnlExactLinearDiffusion< Dimensions > ExactOperator;
   typedef tnlLinearDiffusion< MeshType, Real, Index > ApproximateOperator;

   tnlLinearDiffusionTester(){};

   virtual
   ~tnlLinearDiffusionTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlLinearDiffusionTester" );
      CppUnit :: TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "approximationTest", &TesterType::approximationTest ) );

      return suiteOfTests;
   }

   void expBumpApproximationTest()
   {
      TestFunction testFunction;
      TestFunctionSetter::setTestFunction( testFunction );

      MeshType mesh;
      TestSetter::setMesh( mesh, coarseMeshSize );

      Real l1Err, l2Err, maxErr;
      tnlApproximationError( mesh,
                             exactOperator,
                             approximateOperator,
                             testFunction,
                             l1Err,
                             l2Err,
                             maxErr );

   }


#endif /* TNLLINEARDIFFUSIONTESTER_H_ */
