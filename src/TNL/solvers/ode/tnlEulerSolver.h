/***************************************************************************
                          tnlEulerSolver.h  -  description
                             -------------------
    begin                : 2008/04/01
    copyright            : (C) 2008 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <math.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/solvers/ode/tnlExplicitSolver.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Timer.h>

namespace TNL {

template< typename Problem >
class tnlEulerSolver : public tnlExplicitSolver< Problem >
{
   public:

   typedef Problem  ProblemType;
   typedef typename Problem :: DofVectorType DofVectorType;
   typedef typename Problem :: RealType RealType;
   typedef typename Problem :: DeviceType DeviceType;
   typedef typename Problem :: IndexType IndexType;


   tnlEulerSolver();

   String getType() const;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   void setCFLCondition( const RealType& cfl );

   const RealType& getCFLCondition() const;

   bool solve( DofVectorType& u );

   protected:
   void computeNewTimeLevel( DofVectorType& u,
                             RealType tau,
                             RealType& currentResidue );

 
   DofVectorType k1;

   RealType cflCondition;
 
   //tnlTimer timer, updateTimer;
};

} // namespace TNL

#include <TNL/solvers/ode/tnlEulerSolver_impl.h>
