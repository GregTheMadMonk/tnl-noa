/***************************************************************************
                          tnlExplicitTimeStepper.h  -  description
                             -------------------
    begin                : Jan 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/solvers/ode/tnlODESolverMonitor.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Timer.h>
#include <TNL/Logger.h>

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

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

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
 
   bool writeEpilog( Logger& logger );

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

#include <TNL/solvers/pde/tnlExplicitTimeStepper_impl.h>

