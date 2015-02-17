/***************************************************************************
                          tnlExplicitINSTimeStepper.h  -  description
                             -------------------
    begin                : Feb 17, 2015
    copyright            : (C) 2015 by oberhuber
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

#ifndef EXAMPLES_INCOMPRESSIBLE_NAVIER_STOKES_TNLEXPLICITINSTIMESTEPPER_H_
#define EXAMPLES_INCOMPRESSIBLE_NAVIER_STOKES_TNLEXPLICITINSTIMESTEPPER_H_

template< typename Problem,
          typename LinearSolver >
class tnlExplicitINSTimeStepper
{
   public:

   typedef Problem ProblemType;
   typedef typename Problem::RealType RealType;
   typedef typename Problem::DeviceType DeviceType;
   typedef typename Problem::IndexType IndexType;
   typedef typename Problem::MeshType MeshType;
   typedef typename ProblemType::DofVectorType DofVectorType;

   tnlExplicitINSTimeStepper();

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix = "" );

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

   bool init( const MeshType& mesh );

   void setProblem( ProblemType& problem );

   ProblemType* getProblem() const;

   bool setTimeStep( const RealType& timeStep );

   const RealType& getTimeStep() const;

   bool solve( const RealType& time,
               const RealType& stopTime,
               const MeshType& mesh,
               DofVectorType& dofVector,
               DofVectorType& auxiliaryDofVector );

   protected:

   Problem* problem;

   RealType timeStep;

};

#include "tnlExplicitINSTimeStepper_impl.h"

#endif /* EXAMPLES_INCOMPRESSIBLE_NAVIER_STOKES_TNLEXPLICITINSTIMESTEPPER_H_ */
