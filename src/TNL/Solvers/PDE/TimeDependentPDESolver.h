// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Logger.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Solvers/PDE/PDESolver.h>
#include <TNL/Solvers/PDE/MeshDependentTimeSteps.h>

#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>

namespace TNL {
namespace Solvers {
namespace PDE {

template< typename Problem, typename TimeStepper >
class TimeDependentPDESolver : public PDESolver< typename Problem::RealType, typename Problem::IndexType >,
                               public MeshDependentTimeSteps< typename Problem::MeshType, typename TimeStepper::RealType >
{
public:
   using RealType = typename Problem::RealType;
   using DeviceType = typename Problem::DeviceType;
   using IndexType = typename Problem::IndexType;
   using BaseType = PDESolver< RealType, IndexType >;
   using ProblemType = Problem;
   using MeshType = typename ProblemType::MeshType;
   using DofVectorType = typename ProblemType::DofVectorType;
   using CommonDataType = typename ProblemType::CommonDataType;
   using CommonDataPointer = typename ProblemType::CommonDataPointer;
   using MeshPointer = Pointers::SharedPointer< MeshType, DeviceType >;
   using DofVectorPointer = Pointers::SharedPointer< DofVectorType, DeviceType >;
   using SolverMonitorType = IterativeSolverMonitor< typename Problem::RealType, typename Problem::IndexType >;

   static_assert( ProblemType::isTimeDependent(), "The problem is not time dependent." );

   TimeDependentPDESolver();

   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   bool
   writeProlog( Logger& logger, const Config::ParameterContainer& parameters );

   void
   setProblem( ProblemType& problem );

   void
   setInitialTime( const RealType& initialT );

   const RealType&
   getInitialTime() const;

   bool
   setFinalTime( const RealType& finalT );

   const RealType&
   getFinalTime() const;

   bool
   setTimeStep( const RealType& timeStep );

   const RealType&
   getTimeStep() const;

   bool
   setSnapshotPeriod( const RealType& period );

   const RealType&
   getSnapshotPeriod() const;

   bool
   solve();

   bool
   writeEpilog( Logger& logger ) const;

protected:
   MeshPointer meshPointer;

   Pointers::SharedPointer< Meshes::DistributedMeshes::DistributedMesh< MeshType > > distributedMeshPointer;

   DofVectorType dofs;

   CommonDataPointer commonDataPointer;

   TimeStepper timeStepper;

   ProblemType* problem;

   RealType initialTime, finalTime, snapshotPeriod, timeStep;
};

}  // namespace PDE
}  // namespace Solvers
}  // namespace TNL

#include <TNL/Solvers/PDE/TimeDependentPDESolver.hpp>
