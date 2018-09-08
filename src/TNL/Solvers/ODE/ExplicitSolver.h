/***************************************************************************
                          ExplicitSolver.h  -  description
                             -------------------
    begin                : 2007/06/17
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iomanip>
#include <TNL/Timer.h>
#include <TNL/Experimental/Arithmetics/FlopsCounter.h>
#include <TNL/Object.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>
#include <TNL/Solvers/IterativeSolver.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Containers/Vector.h>

namespace TNL {
namespace Solvers {
namespace ODE {

template< class Problem >
class ExplicitSolver : public IterativeSolver< typename Problem::RealType,
                                                     typename Problem::IndexType >
{
   public:
 
   typedef Problem ProblemType;
   typedef typename Problem :: DofVectorType DofVectorType;
   typedef typename Problem :: RealType RealType;
   typedef typename Problem :: DeviceType DeviceType;
   typedef typename Problem :: IndexType IndexType;
   typedef Pointers::SharedPointer<  DofVectorType, DeviceType > DofVectorPointer;
   typedef IterativeSolverMonitor< RealType, IndexType > SolverMonitorType;

   ExplicitSolver();

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   void setProblem( Problem& problem );

   void setTime( const RealType& t );

   const RealType& getTime() const;
 
   void setStopTime( const RealType& stopTime );

   RealType getStopTime() const;

   void setTau( const RealType& tau );
 
   const RealType& getTau() const;

   void setMaxTau( const RealType& maxTau );

   const RealType& getMaxTau() const;
 
   void setVerbose( IndexType v );

   void setTimer( Timer* timer );

   virtual bool solve( DofVectorPointer& u ) = 0;

   void setTestingMode( bool testingMode );

   void setRefreshRate( const IndexType& refreshRate );

   void refreshSolverMonitor( bool force = false );

protected:
 
   /****
    * Current time of the parabolic problem.
    */
   RealType time;

   /****
    * The method solve will stop when reaching the stopTime.
    */
   RealType stopTime;

   /****
    * Current time step.
    */
   RealType tau;

   RealType maxTau;

   IndexType verbosity;

   Timer* timer;
 
   bool testingMode;

   Problem* problem;

   /****
    * Auxiliary array for the computation of the solver residue on CUDA device.
    */
   Containers::Vector< RealType, DeviceType, IndexType > cudaBlockResidue;
};

} // namespace ODE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/ODE/ExplicitSolver_impl.h>
