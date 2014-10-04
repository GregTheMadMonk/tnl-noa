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

template< typename Problem,
          typename MatrixSolver >
class tnlSemiImplicitTimeStepper
{
   public:

   typedef Problem ProblemType;
   typedef typename Problem::RealType RealType;
   typedef typename Problem::DeviceType DeviceType;
   typedef typename Problem::IndexType IndexType;
   typedef typename Problem::MeshType MeshType;
   typedef typename ProblemType::DofVectorType DofVectorType;
   typedef MatrixSolver MatrixSolverType;

   tnlSemiImplicitTimeStepper();

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix = "" );

   bool init( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

   void setProblem( ProblemType& problem );

   ProblemType* getProblem() const;

   bool setTau( const RealType& tau );

   const RealType& getTau() const;

   bool solve( const RealType& time,
               const RealType& stopTime,
               const MeshType& mesh,
               DofVectorType& dofVector );

   protected:

   MatrixSolver matrixSolver;

   Problem* problem;

   RealType tau;
};

#include <implementation/solvers/pde/tnlSemiImplicitTimeStepper_impl.h>

#endif /* TNLSEMIIMPLICITTIMESTEPPER_H_ */
