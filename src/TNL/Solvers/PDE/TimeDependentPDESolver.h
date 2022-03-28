// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Config/ConfigDescription.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ParameterContainer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Logger.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Pointers/SharedPointer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/PDE/PDESolver.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/PDE/MeshDependentTimeSteps.h>

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/DistributedMeshes/DistributedMesh.h>


namespace noa::TNL {
namespace Solvers {
namespace PDE {

template< typename Problem,
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
      typedef typename ProblemType::CommonDataType CommonDataType;
      typedef typename ProblemType::CommonDataPointer CommonDataPointer;
      typedef Pointers::SharedPointer< MeshType, DeviceType > MeshPointer;
      typedef Pointers::SharedPointer< DofVectorType, DeviceType > DofVectorPointer;
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

      Pointers::SharedPointer< Meshes::DistributedMeshes::DistributedMesh<MeshType> > distributedMeshPointer;

      DofVectorPointer dofsPointer;

      CommonDataPointer commonDataPointer;

      TimeStepper timeStepper;

      ProblemType* problem;

      RealType initialTime, finalTime, snapshotPeriod, timeStep;
};

} // namespace PDE
} // namespace Solvers
} // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/PDE/TimeDependentPDESolver.hpp>
