/***************************************************************************
                          TimeDependentPDESolver.h  -  description
                             -------------------
    begin                : Jan 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Logger.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Solvers/PDE/PDESolver.h>
#include <TNL/Solvers/PDE/MeshDependentTimeSteps.h>

namespace TNL {
namespace Solvers {
namespace PDE {   

template< typename Problem,
          typename DiscreteSolver,
          typename TimeStepper >
class TimeDependentPDESolver
   : public PDESolver< typename Problem::RealType, 
                       typename Problem::IndexType >,
     public MeshDependentTimeSteps< typename Problem::MeshType,
                                    typename TimeStepper::RealType >
{
   public:

      using RealType = typename Problem::RealType;
      using DeviceType = typename Problem::DeviceType;
      using IndexType = typename Problem::IndexType;
      using BaseType = PDESolver< RealType, IndexType >;
      using ProblemType = Problem;
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

      void setProblem( ProblemType& problem );

      void setInitialTime( const RealType& initialT );

      const RealType& getInitialTime() const;

      bool setFinalTime( const RealType& finalT );

      const RealType& getFinalTime() const;

      bool setTimeStep( const RealType& timeStep );

      const RealType& getTimeStep() const;

      bool setSnapshotPeriod( const RealType& period );

      const RealType& getSnapshotPeriod() const;

      bool solve();

      bool writeEpilog( Logger& logger ) const;

   protected:

      MeshPointer meshPointer;

      DofVectorPointer dofsPointer;

      MeshDependentDataPointer meshDependentDataPointer;

      TimeStepper timeStepper;
      
      DiscreteSolver discreteSolver;
      
      ProblemType* problem;

      RealType initialTime, finalTime, snapshotPeriod, timeStep;
};

} // namespace PDE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/PDE/TimeDependentPDESolver_impl.h>
