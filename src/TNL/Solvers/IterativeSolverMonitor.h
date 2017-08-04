/***************************************************************************
                          IterativeSolverMonitor.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Solvers/SolverMonitor.h>

namespace TNL {
namespace Solvers {   

template< typename Real, typename Index>
class IterativeSolverMonitor : public SolverMonitor
{
public:
   typedef Index IndexType;
   typedef Real RealType;

   IterativeSolverMonitor();

   void setTime( const RealType& time );

   void setTimeStep( const RealType& timeStep );

   void setStage( const std::string& stage );

   void setIterations( const IndexType& iterations );

   void setResidue( const RealType& residue );

   void setVerbose( const Index& verbose );
 
   virtual void refresh( bool force = false );

protected:
   int getLineWidth();

   RealType time;

   RealType timeStep;

   std::string stage;

   IndexType iterations;

   RealType residue;

   IndexType verbose;
};

} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/IterativeSolverMonitor_impl.h>
