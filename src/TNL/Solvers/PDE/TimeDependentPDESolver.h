/***************************************************************************
                          TimeDependentPDESolver.h  -  description
                             -------------------
    begin                : Jan 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Object.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Logger.h>
#include <TNL/Timer.h>
#include <TNL/SharedPointer.h>

namespace TNL {
namespace Solvers {
namespace PDE {   

template< typename Problem,
          typename DiscreteSolver,
          typename TimeStepper >
class TimeDependentPDESolver : public Object
{
   public:

      typedef typename TimeStepper::RealType RealType;
      typedef typename TimeStepper::DeviceType DeviceType;
      typedef typename TimeStepper::IndexType IndexType;
      typedef Problem ProblemType;
      typedef typename ProblemType::MeshType MeshType;
      typedef typename ProblemType::DofVectorType DofVectorType;
      typedef typename ProblemType::MeshDependentDataType MeshDependentDataType;
      typedef SharedPointer< MeshType, DeviceType > MeshPointer;
      typedef SharedPointer< DofVectorType, DeviceType > DofVectorPointer;
      typedef SharedPointer< MeshDependentDataType, DeviceType > MeshDependentDataPointer;
      typedef IterativeSolverMonitor< typename Problem::RealType, typename Problem::IndexType > SolverMonitorType;
      
      static_assert( ProblemType::isTimeDependent(), "The problem is not time dependent." );

      TimeDependentPDESolver();

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

      bool writeProlog( Logger& logger,
                        const Config::ParameterContainer& parameters );

      //void setTimeStepper( TimeStepper& timeStepper );

      void setProblem( ProblemType& problem );

      void setInitialTime( const RealType& initialT );

      const RealType& getInitialTime() const;

      bool setFinalTime( const RealType& finalT );

      const RealType& getFinalTime() const;

      bool setTimeStep( const RealType& timeStep );

      const RealType& getTimeStep() const;

      bool setTimeStepOrder( const RealType& timeStepOrder );

      const RealType& getTimeStepOrder() const;

      bool setSnapshotPeriod( const RealType& period );

      const RealType& getSnapshotPeriod() const;

      void setIoTimer( Timer& ioTimer);

      void setComputeTimer( Timer& computeTimer );

      bool solve();

      bool writeEpilog( Logger& logger ) const;

   protected:

      MeshPointer meshPointer;

      DofVectorPointer dofsPointer;

      MeshDependentDataPointer meshDependentDataPointer;

      TimeStepper timeStepper;
      
      DiscreteSolver discreteSolver;

      RealType initialTime, finalTime, snapshotPeriod, timeStep, timeStepOrder;

      ProblemType* problem;

      Timer *ioTimer, *computeTimer;
      
      SolverMonitorType solverMonitor;
      
      SolverMonitor *solverMonitorPointer;
};

} // namespace PDE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/PDE/TimeDependentPDESolver_impl.h>
