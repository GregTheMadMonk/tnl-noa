/***************************************************************************
                          tnlSemiImplicitTimeStepper.h  -  description
                             -------------------
    begin                : Oct 4, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/TimerRT.h>
#include <TNL/Logger.h>

namespace TNL {

template< typename Problem,
          typename LinearSystemSolver >
class tnlSemiImplicitTimeStepper
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
   typedef tnlSharedPointer< MatrixType, DeviceType > MatrixPointer;
   typedef tnlSharedPointer< DofVectorType, DeviceType > DofVectorPointer;
   typedef tnlSharedPointer< PreconditionerType, DeviceType > PreconditionerPointer;

   tnlSemiImplicitTimeStepper();

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   bool init( const MeshPointer& meshPointer );

   void setProblem( ProblemType& problem );

   ProblemType* getProblem() const;

   void setSolver( LinearSystemSolver& linearSystemSolver );

   LinearSystemSolverType* getSolver() const;

   bool setTimeStep( const RealType& timeStep );

   const RealType& getTimeStep() const;

   bool solve( const RealType& time,
               const RealType& stopTime,
               const MeshPointer& meshPointer,
               DofVectorPointer& dofVectorPointer,
               MeshDependentDataType& meshDependentData );
 
   bool writeEpilog( Logger& logger );

   protected:

   Problem* problem;

   MatrixPointer matrix;

   DofVectorPointer rightHandSidePointer;

   LinearSystemSolver* linearSystemSolver;

   RealType timeStep;

   tnlTimer preIterateTimer, linearSystemAssemblerTimer, preconditionerUpdateTimer, linearSystemSolverTimer, postIterateTimer;
 
   bool verbose;
 
   long long int allIterations;
};

} // namespace TNL

#include <TNL/solvers/pde/tnlSemiImplicitTimeStepper_impl.h>

