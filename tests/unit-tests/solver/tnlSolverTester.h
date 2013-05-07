/***************************************************************************
                          tnlSolverTester.h  -  description
                             -------------------
    begin                : Mar 17, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLSOLVERTESTER_H_
#define TNLSOLVERTESTER_H_

#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <solvers/tnlSolver.h>
#include <solvers/tnlSolverMonitor.h>
#include <mesh/tnlGrid.h>
#include <tnlConfig.h>

template< typename Mesh >
class tnlSolverTesterProblem
{
   public:

   typedef typename Mesh :: RealType RealType;
   typedef typename Mesh :: DeviceType DeviceType;
   typedef typename Mesh :: IndexType IndexType;
   typedef Mesh MeshType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlCSRMatrix< RealType, DeviceType, IndexType > DiscreteSolverMatrixType;
   typedef tnlDummyPreconditioner< RealType, DeviceType, IndexType > DiscreteSolverPreconditioner;

   static tnlString getTypeStatic() { return tnlString( "simpleProblemSolver< " ) + Mesh :: getTypeStatic() + " >"; };

   tnlString getPrologHeader() const { return tnlString( "Simple Problem" ); };

   void writeProlog( tnlLogger& logger,
                     const tnlParameterContainer& parameters ) const { };

   bool init( const tnlParameterContainer& parameters ) { this -> dofVector. setSize( 100 ); return true; };

   bool setInitialCondition( const tnlParameterContainer& parameters ) { return true; };

   bool makeSnapshot( const RealType& time, const IndexType& step ) { return true; };

   tnlSolverMonitor< RealType, IndexType >* getSolverMonitor() { return 0; };

   DofVectorType& getDofVector() { return this -> dofVector;};

   void GetExplicitRHS( const RealType& time,
                        const RealType& tau,
                        DofVectorType& _u,
                        DofVectorType& _fu ){};

   protected:

   DofVectorType dofVector;
};

template< typename SolverStarter >
class tnlSolverTesterSetter
{
   public:
   template< typename RealType,
             typename DeviceType,
             typename IndexType >
   bool run( const tnlParameterContainer& parameters ) const
   {
      int dimensions = parameters. GetParameter< int >( "dimensions" );
      if( dimensions <= 0 || dimensions > 3 )
      {
         cerr << "The problem is not defined for " << dimensions << "dimensions." << endl;
         return false;
      }
      SolverStarter solverStarter;
      if( dimensions == 1 )
      {
         typedef tnlGrid< 1, RealType, DeviceType, IndexType > MeshType;
         return solverStarter. run< tnlSolverTesterProblem< MeshType > >( parameters );
      }
      if( dimensions == 2 )
      {
         typedef tnlGrid< 2, RealType, DeviceType, IndexType > MeshType;
         return solverStarter. run< tnlSolverTesterProblem< MeshType > >( parameters );
      }
      if( dimensions == 3 )
      {
         typedef tnlGrid< 3, RealType, DeviceType, IndexType > MeshType;
         return solverStarter. run< tnlSolverTesterProblem< MeshType > >( parameters );
      };
   }

};

class tnlSolverTester : public CppUnit :: TestCase
{
   public:
   tnlSolverTester(){};

   virtual
   ~tnlSolverTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlSolverTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlSolverTester >(
                               "run",
                               & tnlSolverTester :: run )
                             );

      return suiteOfTests;
   };

   void run()
   {
      int argc( 7 );
      /*const char* argv[]{ "tnlSolverTest",
                          "--verbose","0",
                          "--dimensions", "2",
                          "--time-discretisation", "explicit",
                          "--discrete-solver", "merson",
                          "--snapshot-period", "0.01",
                          "--final-time", "1.0" };
      const char configFile[] = TNL_TESTS_DIRECTORY "/data/tnlSolverTest.cfg.desc";
      tnlSolver< tnlSolverTesterSetter > solver;
      CPPUNIT_ASSERT( solver. run( configFile, argc, const_cast< char** >( argv ) ) );*/
   };
};


#endif /* TNLSOLVERTESTER_H_ */