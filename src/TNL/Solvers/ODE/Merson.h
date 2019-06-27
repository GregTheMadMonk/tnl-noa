/***************************************************************************
                          Merson.h  -  description
                             -------------------
    begin                : 2007/06/16
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <math.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/ODE/ExplicitSolver.h>

namespace TNL {
namespace Solvers {
namespace ODE {   

template< class Problem >
class Merson : public ExplicitSolver< Problem >
{
   public:

   typedef Problem ProblemType;
   typedef typename Problem :: DofVectorType DofVectorType;
   typedef typename Problem :: RealType RealType;
   typedef typename Problem :: DeviceType DeviceType;
   typedef typename Problem :: IndexType IndexType;
   typedef Pointers::SharedPointer<  DofVectorType, DeviceType > DofVectorPointer;

   Merson();

   static String getType();

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

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

   void computeNewTimeLevel( const RealType time,
                             const RealType tau,
                             DofVectorPointer& u,
                             RealType& currentResidue );

   void writeGrids( const DofVectorPointer& u );

   DofVectorPointer _k1, _k2, _k3, _k4, _k5, _kAux;

   /****
    * This controls the accuracy of the solver
    */
   RealType adaptivity;
   
   Containers::Vector< RealType, DeviceType, IndexType > openMPErrorEstimateBuffer;
};

} // namespace ODE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/ODE/Merson_impl.h>
