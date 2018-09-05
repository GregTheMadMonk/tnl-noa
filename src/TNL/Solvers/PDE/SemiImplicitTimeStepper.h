/***************************************************************************
                          SemiImplicitTimeStepper.h  -  description
                             -------------------
    begin                : Oct 4, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <memory>  // std::shared_ptr

#include <TNL/Timer.h>
#include <TNL/Logger.h>
#include <TNL/SharedPointer.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>
#include <TNL/Solvers/Linear/LinearSolver.h>

namespace TNL {
namespace Solvers {
namespace PDE {

template< typename Problem >
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
   typedef typename ProblemType::MatrixType MatrixType;
   typedef SharedPointer< MatrixType, DeviceType > MatrixPointer;
   typedef SharedPointer< DofVectorType, DeviceType > DofVectorPointer;
   typedef IterativeSolverMonitor< RealType, IndexType > SolverMonitorType;

   using LinearSolverType = Linear::LinearSolver< MatrixType >;
   using LinearSolverPointer = std::shared_ptr< LinearSolverType >;
   using PreconditionerType = typename LinearSolverType::PreconditionerType;
   using PreconditionerPointer = std::shared_ptr< PreconditionerType >;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   bool init( const MeshPointer& meshPointer );

   void setProblem( ProblemType& problem );

   ProblemType* getProblem() const;

   void setSolverMonitor( SolverMonitorType& solverMonitor );

   bool setTimeStep( const RealType& timeStep );

   const RealType& getTimeStep() const;

   bool solve( const RealType& time,
               const RealType& stopTime,
               DofVectorPointer& dofVectorPointer );
 
   bool writeEpilog( Logger& logger ) const;

   protected:

   // raw pointers with setters
   Problem* problem = nullptr;
   SolverMonitorType* solverMonitor = nullptr;

   // smart pointers initialized to the default-created objects
   MatrixPointer matrix;
   DofVectorPointer rightHandSidePointer;

   // uninitialized smart pointers (they are initialized in the setup method)
   LinearSolverPointer linearSystemSolver = nullptr;
   PreconditionerPointer preconditioner = nullptr;

   RealType timeStep = 0.0;

   Timer preIterateTimer, linearSystemAssemblerTimer, preconditionerUpdateTimer, linearSystemSolverTimer, postIterateTimer;

   bool verbose = false;

   long long int allIterations = 0;
};

} // namespace PDE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/PDE/SemiImplicitTimeStepper_impl.h>
