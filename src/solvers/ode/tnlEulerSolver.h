/***************************************************************************
                          tnlEulerSolver.h  -  description
                             -------------------
    begin                : 2008/04/01
    copyright            : (C) 2008 by Tomas Oberhuber
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

#ifndef tnlEulerSolverH
#define tnlEulerSolverH

#include <math.h>
#include <config/tnlConfigDescription.h>
#include <solvers/ode/tnlExplicitSolver.h>
#include <config/tnlParameterContainer.h>
#include <core/tnlTimer.h>
#include <core/tnlSharedPointer.h>

template< typename Problem >
class tnlEulerSolver : public tnlExplicitSolver< Problem >
{
   public:

   typedef Problem  ProblemType;
   typedef typename Problem :: DofVectorType DofVectorType;
   typedef typename Problem :: RealType RealType;
   typedef typename Problem :: DeviceType DeviceType;
   typedef typename Problem :: IndexType IndexType;
   typedef tnlSharedPointer< DofVectorType, DeviceType > DofVectorPointer;


   tnlEulerSolver();

   tnlString getType() const;

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix = "" );

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

   void setCFLCondition( const RealType& cfl );

   const RealType& getCFLCondition() const;

   bool solve( DofVectorPointer& u );

   protected:
   void computeNewTimeLevel( DofVectorPointer& u,
                             RealType tau,
                             RealType& currentResidue );

   
   DofVectorPointer k1;

   RealType cflCondition;
   
   //tnlTimer timer, updateTimer;
};

#include <solvers/ode/tnlEulerSolver_impl.h>

#endif
