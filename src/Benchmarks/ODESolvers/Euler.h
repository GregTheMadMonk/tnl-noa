/***************************************************************************
                          Euler.h  -  description
                             -------------------
    begin                : 2008/04/01
    copyright            : (C) 2008 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <math.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/ODE/ExplicitSolver.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Timer.h>

#include "../BLAS/CommonVectorOperations.h"

namespace TNL {
namespace Benchmarks {

template< typename Problem,
          typename SolverMonitor = Solvers::IterativeSolverMonitor< typename Problem::RealType, typename Problem::IndexType > >
class Euler : public Solvers::ODE::ExplicitSolver< Problem, SolverMonitor >
{
   public:

   typedef Problem  ProblemType;
   typedef typename Problem::DofVectorType DofVectorType;
   typedef typename Problem::RealType RealType;
   typedef typename Problem::DeviceType DeviceType;
   typedef typename Problem::IndexType IndexType;
   typedef Pointers::SharedPointer<  DofVectorType, DeviceType > DofVectorPointer;
   using VectorOperations = CommonVectorOperations< DeviceType >;


   Euler();

   static String getType();

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   void setCFLCondition( const RealType& cfl );

   const RealType& getCFLCondition() const;

   bool solve( DofVectorPointer& u );

   protected:
   void computeNewTimeLevel( DofVectorPointer& u,
                             RealType tau,
                             RealType& currentResidue );

   
   DofVectorPointer k1;

   RealType cflCondition;
 
   //Timer timer, updateTimer;
};

} // namespace Benchmarks
} // namespace TNL

#include "Euler.hpp"
