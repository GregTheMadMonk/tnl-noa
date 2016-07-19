/***************************************************************************
                          tnlMersonSolver.h  -  description
                             -------------------
    begin                : 2007/06/16
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <math.h>
#include <TNL/config/tnlConfigDescription.h>
#include <TNL/solvers/ode/tnlExplicitSolver.h>

namespace TNL {

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

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix = "" );

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

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

   /****
    * This controls the accuracy of the solver
    */
   RealType adaptivity;
};

} // namespace TNL

#include <TNL/solvers/ode/tnlMersonSolver_impl.h>
