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

   tnlMersonSolver();

   tnlString getType() const;

   void setAdaptivity( const RealType& a );

   bool solve( DofVectorType& u );

   protected:
   
   //! Compute the Runge-Kutta coefficients
   /****
    * The parameter u is not constant because one often
    * needs to correct u on the boundaries to be able to compute
    * the RHS.
    */
   void computeKFunctions( DofVectorType& u,
                           const RealType& time,
                           RealType tau );

   RealType computeError( const RealType tau );

   void computeNewTimeLevel( DofVectorType& u,
                             RealType tau,
                             RealType& currentResidue );

   void writeGrids( const DofVectorType& u );

   DofVectorType k1, k2, k3, k4, k5, kAux;

   //! This controls the accuracy of the solver
   RealType adaptivity;
};

#include <implementation/solvers/ode/tnlMersonSolver_impl.h>

#endif
