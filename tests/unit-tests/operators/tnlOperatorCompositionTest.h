/***************************************************************************
                          tnlOperatorCompositionTest.h  -  description
                             -------------------
    begin                : Feb 11, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLOPERATORFUNCTIONTEST_H
#define	TNLOPERATORFUNCTIONTEST_H

#include <operators/tnlOperatorComposition.h>
#include <mesh/tnlGrid.h>
#include <functions/tnlExpBumpFunction.h>
#include <functions/tnlConstantFunction.h>
#include <operators/diffusion/tnlLinearDiffusion.h>
#include <operators/tnlNeumannBoundaryConditions.h>
#include "../tnlUnitTestStarter.h"

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>


template< typename Operator >
class tnlOperatorCompositionTest
   : public CppUnit::TestCase
{
   public:
   typedef tnlOperatorCompositionTest< Operator > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;
   typedef Operator OperatorType;
   typedef typename OperatorType::MeshType MeshType;
   typedef typename OperatorType::RealType RealType;
   typedef typename OperatorType::IndexType IndexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef typename MeshType::VertexType VertexType;
   typedef tnlExpBumpFunction< MeshType::getMeshDimensions(), typename MeshType::RealType > TestFunctionType;
   typedef tnlConstantFunction< MeshType::getMeshDimensions(), typename MeshType::RealType > ConstantFunction;
   typedef tnlNeumannBoundaryConditions< MeshType, ConstantFunction > BoundaryConditions;
   typedef tnlOperatorComposition< OperatorType, OperatorType, BoundaryConditions > OperatorComposition;
   typedef tnlMeshFunction< MeshType, MeshType::getMeshDimensions() > MeshFunctionType;
   typedef tnlOperatorFunction< OperatorType, MeshFunctionType, BoundaryConditions > OperatorFunction;
   typedef tnlOperatorFunction< OperatorType, OperatorFunction, BoundaryConditions > OperatorFunction2;

   tnlOperatorCompositionTest(){};

   virtual
   ~tnlOperatorCompositionTest(){};

   static CppUnit::Test* suite()
   {
      CppUnit::TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlOperatorCompositionTest" );
      CppUnit::TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "test", &TesterType::test ) );
      return suiteOfTests;
   }
 
   void test()
   {
      MeshType mesh;
      mesh.setDimensions( CoordinatesType( 25 ) );
      mesh.setDomain( VertexType( -1.0 ), VertexType( 2.0 ) );
      TestFunctionType testFunction;
      testFunction.setAmplitude( 1.0 );
      testFunction.setSigma( 1.0 );
      MeshFunctionType f1( mesh );
      f1 = testFunction;
      OperatorType operator_;
      BoundaryConditions boundaryConditions;
      OperatorFunction operatorFunction1( operator_, boundaryConditions, f1 );
      operatorFunction1.refresh();
      OperatorFunction2 operatorFunction2( operator_, boundaryConditions, operatorFunction1 );
      operatorFunction2.refresh();
 
      //f1 = testFunction;
      OperatorComposition operatorComposition( operator_, operator_, boundaryConditions, mesh );
      //operatorComposition.refresh();
      tnlOperatorFunction< OperatorComposition, MeshFunctionType, BoundaryConditions > operatorFunction3( operatorComposition, boundaryConditions, f1 );
      operatorFunction3.refresh();
 
      /*f1 = testFunction;
      f1.write( "testFunction", "gnuplot" );
      f1 = operatorFunction1;
      f1.write( "operator1", "gnuplot" );
      f1 = operatorFunction2;
      f1.write( "operator2", "gnuplot" );
 
      f1 = operatorFunction3;
      f1.write( "operatorComposition", "gnuplot" );      */
 
      //CPPUNIT_ASSERT( operatorFunction2 == operatorFunction3 );
      for( IndexType i = 0; i < mesh.template getEntitiesCount< typename MeshType::Cell >(); i++ )
      {
         auto entity = mesh.template getEntity< typename MeshType::Cell >( i );
         entity.refresh();
         //cerr << entity.getIndex() << " " << operatorFunction2( entity ) << " " << operatorFunction3( entity ) << endl;
         CPPUNIT_ASSERT( operatorFunction2( entity ) == operatorFunction3( entity ) );
         /*if( entity.isBoundaryEntity() )
            CPPUNIT_ASSERT( boundaryConditions( f1, entity ) == operatorFunction( entity ) );
         else
         {
 
 
         }*/
      }
   }
};
#endif
 
template< typename Operator >
bool runTest()
{
   //tnlOperatorCompositionTest< Operator > test;
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter::run< tnlOperatorCompositionTest< Operator > >() )
     return false;
   return true;
#else
   return false;
#endif
}

template< typename MeshType >
bool setOperator()
{
   return runTest< tnlLinearDiffusion< MeshType > >();
}

int main( int argc, char* argv[] )
{
   if( ! setOperator< tnlGrid< 1, double, tnlHost, int > >() ||
       ! setOperator< tnlGrid< 2, double, tnlHost, int > >() ||
       ! setOperator< tnlGrid< 3, double, tnlHost, int > >() )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

#endif	/* TNLOPERATORFUNCTIONTEST_H */

