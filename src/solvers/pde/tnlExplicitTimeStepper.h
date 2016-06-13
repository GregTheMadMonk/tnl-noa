/***************************************************************************
                          tnlExplicitTimeStepper.h  -  description
                             -------------------
    begin                : Jan 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLEXPLICITTIMESTEPPER_H_
#define TNLEXPLICITTIMESTEPPER_H_

#include <solvers/ode/tnlODESolverMonitor.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <core/tnlTimer.h>
#include <core/tnlLogger.h>
#include <core/tnlSharedPointer.h>

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
   typedef tnlSharedPointer< MeshType > MeshPointer;
   typedef typename ProblemType::DofVectorType DofVectorType;
   typedef typename ProblemType::MeshDependentDataType MeshDependentDataType;
   typedef tnlSharedPointer< DofVectorType, DeviceType > DofVectorPointer;

   tnlExplicitTimeStepper();

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix = "" );

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

   bool init( const MeshPointer& meshPointer );

   void setSolver( OdeSolverType& odeSolver );

   void setProblem( ProblemType& problem );

   ProblemType* getProblem() const;

   bool setTimeStep( const RealType& tau );

   const RealType& getTimeStep() const;

   bool solve( const RealType& time,
               const RealType& stopTime,
               const MeshPointer& mesh,
               DofVectorPointer& dofVector,
               MeshDependentDataType& meshDependentData );

   void getExplicitRHS( const RealType& time,
                        const RealType& tau,
                        DofVectorPointer& _u,
                        DofVectorPointer& _fu );
   
   bool writeEpilog( tnlLogger& logger );

   protected:

   OdeSolverType* odeSolver;

   Problem* problem;

   MeshPointer meshPointer;

   RealType timeStep;

   MeshDependentDataType* meshDependentData;
   
   tnlTimer preIterateTimer, explicitUpdaterTimer, mainTimer, postIterateTimer;
   
   long long int allIterations;
};

#include <solvers/pde/tnlExplicitTimeStepper_impl.h>

#endif /* TNLEXPLICITTIMESTEPPER_H_ */
