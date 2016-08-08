/***************************************************************************
                          ODESolverMonitor.h  -  description
                             -------------------
    begin                : Mar 12, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Solvers/tnlIterativeSolverMonitor.h>

namespace TNL {
namespace Solvers {
namespace ODE {   

template< typename Real, typename Index>
class ODESolverMonitor : public tnlIterativeSolverMonitor< Real, Index >
{
   public:

   typedef Real RealType;
   typedef Index IndexType;

   ODESolverMonitor();

   void refresh();

   void setTimeStep( const RealType& timeStep );

   const RealType& getTimeStep() const;

   void setTime( const RealType& time );

   const RealType& getTime() const;

   protected:

   RealType timeStep;

   RealType time;

};

} // namespace ODE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/ODE/ODESolverMonitor_impl.h>
