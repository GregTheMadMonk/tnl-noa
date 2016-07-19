/***************************************************************************
                          tnlExplicitTimeStepper.h  -  description
                             -------------------
    begin                : Jan 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <solvers/ode/tnlODESolverMonitor.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <core/tnlTimer.h>
#include <core/tnlLogger.h>

namespace TNL {

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
class tnlExplicitTimeStepper
{
   public:

   typedef Problem ProblemType;
   typedef OdeSolver< tnlExplicitTimeStepper< Problem, OdeSolver > > OdeSolverType;
   typedef typename Problem::RealType RealType;
   typedef typename Problem::DeviceType DeviceType;
   typedef typename Problem::IndexType IndexType;
   typedef typename Problem::MeshType MeshType;
   typedef typename ProblemType::DofVectorType DofVectorType;
   typedef typename ProblemType::MeshDependentDataType MeshDependentDataType;

   tnlExplicitTimeStepper();

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix = "" );

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

   bool init( const MeshType& mesh );

   void setSolver( OdeSolverType& odeSolver );

   void setProblem( ProblemType& problem );

   ProblemType* getProblem() const;

   bool setTimeStep( const RealType& tau );

   const RealType& getTimeStep() const;

   bool solve( const RealType& time,
               const RealType& stopTime,
               const MeshType& mesh,
               DofVectorType& dofVector,
               MeshDependentDataType& meshDependentData );

   void getExplicitRHS( const RealType& time,
                        const RealType& tau,
                        DofVectorType& _u,
                        DofVectorType& _fu );
 
   bool writeEpilog( tnlLogger& logger );

   protected:

   OdeSolverType* odeSolver;

   Problem* problem;

   const MeshType* mesh;

   RealType timeStep;

   MeshDependentDataType* meshDependentData;
 
   tnlTimer preIterateTimer, explicitUpdaterTimer, mainTimer, postIterateTimer;
 
   long long int allIterations;
};

} // namespace TNL

#include <solvers/pde/tnlExplicitTimeStepper_impl.h>

