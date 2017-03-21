/***************************************************************************
                          SolverTester.h  -  description
                             -------------------
    begin                : Mar 17, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef SolverTESTER_H_
#define SolverTESTER_H_

#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <TNL/Solvers/Solver.h>
#include <TNL/Solvers/SolverMonitor.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/tnlConfig.h>

using namespace TNL;

template< typename Mesh >
class SolverTesterProblem
{
   public:

   typedef typename Mesh :: RealType RealType;
   typedef typename Mesh :: DeviceType DeviceType;
   typedef typename Mesh :: IndexType IndexType;
   typedef Mesh MeshType;
   typedef Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
   typedef Matrices::CSR< RealType, DeviceType, IndexType > DiscreteSolverMatrixType;
   typedef Solvers::Linear::Preconditioners::Dummy< RealType, DeviceType, IndexType > DiscreteSolverPreconditioner;

   static String getTypeStatic() { return String( "simpleProblemSolver< " ) + Mesh :: getTypeStatic() + " >"; };

   String getPrologHeader() const { return String( "Simple Problem" ); };

   void writeProlog( Logger& logger,
                     const Config::ParameterContainer& parameters ) const { };

   bool setup( const Config::ParameterContainer& parameters ) { this->dofVector. setSize( 100 ); return true; };

   bool setInitialCondition( const Config::ParameterContainer& parameters ) { return true; };

   bool makeSnapshot( const RealType& time, const IndexType& step ) { return true; };

   Solvers::SolverMonitor< RealType, IndexType >* getSolverMonitor() { return 0; };

   DofVectorType& getDofVector() { return this->dofVector;};

   void getExplicitUpdate( const RealType& time,
                        const RealType& tau,
                        DofVectorType& _u,
                        DofVectorType& _fu ){};

   protected:

   DofVectorType dofVector;
};

template< typename SolverStarter >
class SolverTesterSetter
{
   public:
   template< typename RealType,
             typename DeviceType,
             typename IndexType >
   bool run( const Config::ParameterContainer& parameters ) const
   {
      int dimensions = parameters. getParameter< int >( "dimensions" );
      if( dimensions <= 0 || dimensions > 3 )
      {
         std::cerr << "The problem is not defined for " << dimensions << "dimensions." << std::endl;
         return false;
      }
      SolverStarter solverStarter;
      if( dimensions == 1 )
      {
         typedef Meshes::Grid< 1, RealType, DeviceType, IndexType > MeshType;
         return solverStarter. run< SolverTesterProblem< MeshType > >( parameters );
      }
      if( dimensions == 2 )
      {
         typedef Meshes::Grid< 2, RealType, DeviceType, IndexType > MeshType;
         return solverStarter. run< SolverTesterProblem< MeshType > >( parameters );
      }
      if( dimensions == 3 )
      {
         typedef Meshes::Grid< 3, RealType, DeviceType, IndexType > MeshType;
         return solverStarter. run< SolverTesterProblem< MeshType > >( parameters );
      };
   }

};

class SolverTester : public CppUnit :: TestCase
{
   public:
   SolverTester(){};

   virtual
   ~SolverTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "SolverTester" );
      CppUnit :: TestResult result;
      suiteOfTests -> addTest( new CppUnit :: TestCaller< SolverTester >(
                               "run",
                               & SolverTester :: run )
                             );

      return suiteOfTests;
   };

   void run()
   {
      int argc( 7 );
      /*const char* argv[]{ "SolverTest",
                          "--verbose","0",
                          "--dimensions", "2",
                          "--time-discretisation", "explicit",
                          "--discrete-solver", "merson",
                          "--snapshot-period", "0.01",
                          "--final-time", "1.0" };
      const char configFile[] = TNL_TESTS_DIRECTORY "/data/SolverTest.cfg.desc";
      Solver< SolverTesterSetter > solver;
      CPPUNIT_ASSERT( solver. run( configFile, argc, const_cast< char** >( argv ) ) );*/
   };
};


#endif /* SolverTESTER_H_ */
