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
#include <TNL/TimerCPU.h>
#include <TNL/TimerRT.h>
#include <TNL/core/tnlFlopsCounter.h>
#include <TNL/Object.h>
#include <TNL/Solvers/ODE/ODESolverMonitor.h>
#include <TNL/Solvers/tnlIterativeSolver.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/tnlSharedPointer.h>

namespace TNL {
namespace Solvers {
namespace ODE {

template< class Problem >
class ExplicitSolver : public tnlIterativeSolver< typename Problem::RealType,
                                                     typename Problem::IndexType >
{
   public:
 
   typedef Problem ProblemType;
   typedef typename Problem :: DofVectorType DofVectorType;
   typedef typename Problem :: RealType RealType;
   typedef typename Problem :: DeviceType DeviceType;
   typedef typename Problem :: IndexType IndexType;
   typedef tnlSharedPointer< DofVectorType, DeviceType >  DofVectorPointer;

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

   void setMPIComm( MPI_Comm comm );
 
   void setVerbose( IndexType v );

   void setTimerCPU( TimerCPU* timer );

   void setTimerRT( TimerRT* timer );
   
   virtual bool solve( DofVectorPointer& u ) = 0;

   void setTestingMode( bool testingMode );

   void setRefreshRate( const IndexType& refreshRate );

   void setSolverMonitor( ODESolverMonitor< RealType, IndexType >& solverMonitor );

   void refreshSolverMonitor();

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

   MPI_Comm solver_comm;

   IndexType verbosity;

   TimerCPU* cpu_timer;
 
   TimerRT* rt_timer;

   bool testingMode;

   Problem* problem;

   ODESolverMonitor< RealType, IndexType >* solverMonitor;

   /****
    * Auxiliary array for the computation of the solver residue on CUDA device.
    */
   Vectors::Vector< RealType, DeviceType, IndexType > cudaBlockResidue;
};

} // namespace ODE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/ODE/ExplicitSolver_impl.h>
