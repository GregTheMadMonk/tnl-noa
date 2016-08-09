/***************************************************************************
                          OperatorCompositionTest.h  -  description
                             -------------------
    begin                : Feb 11, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef OperatorFunctionTEST_H
#define	OperatorFunctionTEST_H

#include <TNL/Operators/OperatorComposition.h>
#include <TNL/mesh/tnlGrid.h>
#include <TNL/Functions/Analytic/ExpBumpFunction.h>
#include <TNL/Functions/Analytic/ConstantFunction.h>
#include <TNL/Operators/diffusion/LinearDiffusion.h>
#include <TNL/Operators/NeumannBoundaryConditions.h>
#include "../tnlUnitTestStarter.h"

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>

using namespace TNL;

template< typename Operator >
class OperatorCompositionTest
   : public CppUnit::TestCase
{
   public:
   typedef OperatorCompositionTest< Operator > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;
   typedef Operator OperatorType;
   typedef typename OperatorType::MeshType MeshType;
   typedef typename OperatorType::RealType RealType;
   typedef typename OperatorType::IndexType IndexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef typename MeshType::VertexType VertexType;
   typedef Functions::Analytic::ExpBumpFunction< MeshType::getMeshDimensions(), typename MeshType::RealType > TestFunctionType;
   typedef Functions::Analytic::ConstantFunction< MeshType::getMeshDimensions(), typename MeshType::RealType > ConstantFunction;
   typedef Operators::NeumannBoundaryConditions< MeshType, ConstantFunction > BoundaryConditions;
   typedef Operators::OperatorComposition< OperatorType, OperatorType, BoundaryConditions > OperatorComposition;
   typedef Functions::MeshFunction< MeshType, MeshType::getMeshDimensions() > MeshFunctionType;
   typedef Functions::OperatorFunction< OperatorType, MeshFunctionType, BoundaryConditions > OperatorFunction;
   typedef Functions::OperatorFunction< OperatorType, OperatorFunction, BoundaryConditions > OperatorFunction2;

   OperatorCompositionTest(){};

   virtual
   ~OperatorCompositionTest(){};

   static CppUnit::Test* suite()
   {
      CppUnit::TestSuite* suiteOfTests = new CppUnit :: TestSuite( "OperatorCompositionTest" );
      CppUnit::TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "test", &TesterType::test ) );
      return suiteOfTests;
   }
 
   void test()
   {      
      SharedPointer< MeshType > mesh;
      mesh->setDimensions( CoordinatesType( 25 ) );
      mesh->setDomain( VertexType( -1.0 ), VertexType( 2.0 ) );
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
      Functions::OperatorFunction< OperatorComposition, MeshFunctionType, BoundaryConditions > operatorFunction3( operatorComposition, boundaryConditions, f1 );
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
      for( IndexType i = 0; i < mesh->template getEntitiesCount< typename MeshType::Cell >(); i++ )
      {
         auto entity = mesh->template getEntity< typename MeshType::Cell >( i );
         entity.refresh();
         //cerr << entity.getIndex() << " " << operatorFunction2( entity ) << " " << operatorFunction3( entity ) << std::endl;
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
   //OperatorCompositionTest< Operator > test;
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter::run< OperatorCompositionTest< Operator > >() )
     return false;
   return true;
#else
   return false;
#endif
}

using namespace TNL;

template< typename MeshType >
bool setOperator()
{
   return runTest< Operators::LinearDiffusion< MeshType > >();
}

int main( int argc, char* argv[] )
{
   if( ! setOperator< tnlGrid< 1, double, Devices::Host, int > >() ||
       ! setOperator< tnlGrid< 2, double, Devices::Host, int > >() ||
       ! setOperator< tnlGrid< 3, double, Devices::Host, int > >() )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

#endif	/* OperatorFunctionTEST_H */

