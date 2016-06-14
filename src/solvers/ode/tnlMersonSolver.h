/***************************************************************************
                          tnlMersonSolver.h  -  description
                             -------------------
    begin                : 2007/06/16
    copyright            : (C) 2007 by Tomas Oberhuber
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

#ifndef tnlMersonSolverH
#define tnlMersonSolverH

#include <math.h>
#include <config/tnlConfigDescription.h>
#include <solvers/ode/tnlExplicitSolver.h>

template< class Problem >
class tnlMersonSolver : public tnlExplicitSolver< Problem >
{
   public:

   typedef Problem ProblemType;
   typedef typename Problem :: DofVectorType DofVectorType;
   typedef typename Problem :: RealType RealType;
   typedef typename Problem :: DeviceType DeviceType;
   typedef typename Problem :: IndexType IndexType;
   typedef tnlSharedPointer< DofVectorType, DeviceType > DofVectorPointer;

   tnlMersonSolver();

   tnlString getType() const;

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix = "" );

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

   void setAdaptivity( const RealType& a );

   bool solve( DofVectorPointer& u );

   protected:
   
   //! Compute the Runge-Kutta coefficients
   /****
    * The parameter u is not constant because one often
    * needs to correct u on the boundaries to be able to compute
    * the RHS.
    */
   void computeKFunctions( DofVectorPointer& u,
                           const RealType& time,
                           RealType tau );

   RealType computeError( const RealType tau );

   void computeNewTimeLevel( DofVectorPointer& u,
                             RealType tau,
                             RealType& currentResidue );

   void writeGrids( const DofVectorPointer& u );

   DofVectorPointer k1, k2, k3, k4, k5, kAux;

   /****
    * This controls the accuracy of the solver
    */
   RealType adaptivity;
};

#include <solvers/ode/tnlMersonSolver_impl.h>

#endif
