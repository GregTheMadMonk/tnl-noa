/***************************************************************************
                          SemiImplicitTimeStepper.h  -  description
                             -------------------
    begin                : Oct 4, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Timer.h>
#include <TNL/Logger.h>
#include <TNL/SharedPointer.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>

namespace TNL {
namespace Solvers {
namespace PDE {   

template< typename Problem,
          typename LinearSystemSolver >
class SemiImplicitTimeStepper
{
   public:

   typedef Problem ProblemType;
   typedef typename Problem::RealType RealType;
   typedef typename Problem::DeviceType DeviceType;
   typedef typename Problem::IndexType IndexType;
   typedef typename Problem::MeshType MeshType;
   typedef typename Problem::MeshPointer MeshPointer;
   typedef typename ProblemType::DofVectorType DofVectorType;   
   typedef typename ProblemType::MeshDependentDataType MeshDependentDataType;
   typedef LinearSystemSolver LinearSystemSolverType;
   typedef typename LinearSystemSolverType::PreconditionerType PreconditionerType;
   typedef typename ProblemType::MatrixType MatrixType;
   typedef SharedPointer< MatrixType, DeviceType > MatrixPointer;
   typedef SharedPointer< DofVectorType, DeviceType > DofVectorPointer;
   typedef SharedPointer< MeshDependentDataType, DeviceType > MeshDependentDataPointer;
   typedef SharedPointer< PreconditionerType, DeviceType > PreconditionerPointer;
   typedef IterativeSolverMonitor< RealType, IndexType > SolverMonitorType;

   SemiImplicitTimeStepper();

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   bool init( const MeshPointer& meshPointer );

   void setProblem( ProblemType& problem );

   ProblemType* getProblem() const;

   void setSolver( LinearSystemSolver& linearSystemSolver );

   void setSolverMonitor( SolverMonitorType& solverMonitor );

   LinearSystemSolverType* getSolver() const;

   bool setTimeStep( const RealType& timeStep );

   const RealType& getTimeStep() const;

   bool solve( const RealType& time,
               const RealType& stopTime,
               const MeshPointer& meshPointer,
               DofVectorPointer& dofVectorPointer,
               MeshDependentDataPointer& meshDependentData );
 
   bool writeEpilog( Logger& logger );

   protected:

   Problem* problem;

   MatrixPointer matrix;

   DofVectorPointer rightHandSidePointer;

   LinearSystemSolver* linearSystemSolver;

   SolverMonitorType* solverMonitor;

   RealType timeStep;

   Timer preIterateTimer, linearSystemAssemblerTimer, preconditionerUpdateTimer, linearSystemSolverTimer, postIterateTimer;
 
   bool verbose;
 
   long long int allIterations;
};

} // namespace PDE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/PDE/SemiImplicitTimeStepper_impl.h>

