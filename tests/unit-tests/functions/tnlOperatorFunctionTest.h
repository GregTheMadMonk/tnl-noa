/***************************************************************************
                          OperatorFunctionTest.h  -  description
                             -------------------
    begin                : Feb 11, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef OperatorFunctionTEST_H
#define	OperatorFunctionTEST_H

#include <TNL/Functions/OperatorFunction.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/Analytic/ExpBump.h>
#include <TNL/Operators/diffusion/LinearDiffusion.h>
#include <TNL/Operators/DirichletBoundaryConditions.h>
#include "../tnlUnitTestStarter.h"

#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>

using namespace TNL;

template< typename Operator,
          bool EvaluateOnFly >
class OperatorFunctionTest
   : public CppUnit::TestCase
{
   public:
   typedef OperatorFunctionTest< Operator, EvaluateOnFly > TesterType;
   typedef typename CppUnit::TestCaller< TesterType > TestCallerType;
   typedef Operator OperatorType;
   typedef typename OperatorType::MeshType MeshType;
   typedef typename OperatorType::RealType RealType;
   typedef typename OperatorType::IndexType IndexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef typename MeshType::PointType PointType;
   typedef Functions::Analytic::ExpBump< MeshType::getMeshDimension(), RealType > TestFunctionType;
   typedef Functions::MeshFunction< MeshType, MeshType::getMeshDimension() > MeshFunctionType;
   typedef SharedPointer< MeshType > MeshPointer;

   OperatorFunctionTest(){};

   virtual
   ~OperatorFunctionTest(){};

   static CppUnit::Test* suite()
   {
      CppUnit::TestSuite* suiteOfTests = new CppUnit :: TestSuite( "OperatorFunctionTest" );
      CppUnit::TestResult result;

      suiteOfTests -> addTest( new TestCallerType( "testWithNoBoundaryConditions", &TesterType::testWithNoBoundaryConditions ) );
      suiteOfTests -> addTest( new TestCallerType( "testWithBoundaryConditions", &TesterType::testWithBoundaryConditions ) );
      return suiteOfTests;
   }
 
   void testWithNoBoundaryConditions()
   {
      MeshPointer meshPointer;
      typedef Functions::OperatorFunction< Operator, MeshFunctionType, void, EvaluateOnFly > OperatorFunctionType;
      meshPointer->setDimensions( CoordinatesType( 25 ) );
      meshPointer->setDomain( PointType( -1.0 ), PointType( 2.0 ) );
      TestFunctionType testFunction;
      testFunction.setAmplitude( 1.0 );
      testFunction.setSigma( 1.0 );
      MeshFunctionType f1( meshPointer );
      f1 = testFunction;
      OperatorType operator_;
      OperatorFunctionType operatorFunction( operator_, f1 );
      operatorFunction.refresh();
      //cerr << f1.getData() << endl;
      for( IndexType i = 0; i < meshPointer->template getEntitiesCount< typename MeshType::Cell >(); i++ )
      {
         auto entity = meshPointer->template getEntity< typename MeshType::Cell >( i );
         entity.refresh();
 
         if( ! entity.isBoundaryEntity() )
         {
            //cerr << entity.getIndex() << " " << operator_( f1, entity ) << " " << operatorFunction( entity ) << std::endl;
            CPPUNIT_ASSERT( operator_( f1, entity ) == operatorFunction( entity ) );
         }
      }
   }
 
   void testWithBoundaryConditions()
   {
      SharedPointer< MeshType > mesh;
      typedef Operators::DirichletBoundaryConditions< MeshType > BoundaryConditionsType;
      typedef Functions::OperatorFunction< Operator, MeshFunctionType, BoundaryConditionsType, EvaluateOnFly > OperatorFunctionType;
      mesh->setDimensions( CoordinatesType( 25 ) );
      mesh->setDomain( PointType( -1.0 ), PointType( 2.0 ) );
      TestFunctionType testFunction;
      testFunction.setAmplitude( 1.0 );
      testFunction.setSigma( 1.0 );
      MeshFunctionType f1( mesh );
      f1 = testFunction;
      OperatorType operator_;
      BoundaryConditionsType boundaryConditions;
      OperatorFunctionType operatorFunction( operator_, boundaryConditions, f1 );
      operatorFunction.refresh();
      //cerr << f1.getData() << endl;
      for( IndexType i = 0; i < mesh->template getEntitiesCount< typename MeshType::Cell >(); i++ )
      {
         auto entity = mesh->template getEntity< typename MeshType::Cell >( i );
         entity.refresh();
         if( entity.isBoundaryEntity() )
            CPPUNIT_ASSERT( boundaryConditions( f1, entity ) == operatorFunction( entity ) );
         else
         {
            //cerr << entity.getIndex() << " " << operator_( f1, entity ) << " " << operatorFunction( entity ) << std::endl;
            CPPUNIT_ASSERT( operator_( f1, entity ) == operatorFunction( entity ) );
         }
      }
   }
};
#endif
 
template< typename MeshType >
bool runTest()
{
#ifdef HAVE_CPPUNIT
   typedef Operators::LinearDiffusion< MeshType > OperatorType;
   OperatorFunctionTest< OperatorType, false > test;
   //test.testWithBoundaryConditions();
   if( //! tnlUnitTestStarter::run< OperatorFunctionTest< OperatorType, true > >() ||
       ! tnlUnitTestStarter::run< OperatorFunctionTest< OperatorType, false > >() )
     return false;
   return true;
#else
   return false;
#endif
}

int main( int argc, char* argv[] )
{
   using namespace TNL;
   if( ! runTest< Meshes::Grid< 1, double, Devices::Host, int > >() ||
       ! runTest< Meshes::Grid< 2, double, Devices::Host, int > >() ||
       ! runTest< Meshes::Grid< 3, double, Devices::Host, int > >() )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

#endif	/* OperatorFunctionTEST_H */

