/***************************************************************************
                          ExplicitTimeStepper.h  -  description
                             -------------------
    begin                : Jan 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Timer.h>
#include <TNL/Logger.h>
#include <TNL/SharedPointer.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>

namespace TNL {
namespace Solvers {
namespace PDE {   

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
class ExplicitTimeStepper
{
   public:

   typedef Problem ProblemType;
   typedef OdeSolver< ExplicitTimeStepper< Problem, OdeSolver > > OdeSolverType;
   typedef typename Problem::RealType RealType;
   typedef typename Problem::DeviceType DeviceType;
   typedef typename Problem::IndexType IndexType;
   typedef typename Problem::MeshType MeshType;
   typedef SharedPointer< MeshType > MeshPointer;
   typedef typename ProblemType::DofVectorType DofVectorType;
   typedef typename ProblemType::MeshDependentDataType MeshDependentDataType;
   typedef SharedPointer< DofVectorType, DeviceType > DofVectorPointer;
   typedef SharedPointer< MeshDependentDataType, DeviceType > MeshDependentDataPointer;
   typedef IterativeSolverMonitor< RealType, IndexType > SolverMonitorType;

   ExplicitTimeStepper();

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   bool init( const MeshPointer& meshPointer );

   void setSolver( OdeSolverType& odeSolver );

   void setSolverMonitor( SolverMonitorType& solverMonitor );

   void setProblem( ProblemType& problem );

   ProblemType* getProblem() const;

   bool setTimeStep( const RealType& tau );

   const RealType& getTimeStep() const;

   bool solve( const RealType& time,
               const RealType& stopTime,
               const MeshPointer& mesh,
               DofVectorPointer& dofVector,
               MeshDependentDataPointer& meshDependentData );

   void getExplicitRHS( const RealType& time,
                        const RealType& tau,
                        DofVectorPointer& _u,
                        DofVectorPointer& _fu );
   
   bool writeEpilog( Logger& logger );

   protected:

   OdeSolverType* odeSolver;

   SolverMonitorType* solverMonitor;

   Problem* problem;

   MeshPointer mesh;

   RealType timeStep;

   MeshDependentDataPointer meshDependentData;
 
   Timer preIterateTimer, explicitUpdaterTimer, mainTimer, postIterateTimer;
 
   long long int allIterations;
};

} // namespace PDE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/PDE/ExplicitTimeStepper_impl.h>

