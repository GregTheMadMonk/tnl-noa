/***************************************************************************
                          tnlSemiImplicitTimeStepper.h  -  description
                             -------------------
    begin                : Oct 4, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLSEMIIMPLICITTIMESTEPPER_H_
#define TNLSEMIIMPLICITTIMESTEPPER_H_

#include <core/tnlTimerRT.h>
#include <core/tnlLogger.h>

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

   tnlSemiImplicitTimeStepper();

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix = "" );

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

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
               DofVectorType& dofVector,
               MeshDependentDataType& meshDependentData );
   
   bool writeEpilog( tnlLogger& logger );

   protected:

   Problem* problem;

   MatrixType matrix;

   DofVectorType rightHandSide;

   LinearSystemSolver* linearSystemSolver;

   RealType timeStep;

   tnlTimer preIterateTimer, linearSystemAssemblerTimer, linearSystemSolverTimer, postIterateTimer;
   
   bool verbose;
   
   long long int allIterations;
};

#include <solvers/pde/tnlSemiImplicitTimeStepper_impl.h>

#endif /* TNLSEMIIMPLICITTIMESTEPPER_H_ */
