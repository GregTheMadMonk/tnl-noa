/***************************************************************************
                          tnlODESolverMonitor.h  -  description
                             -------------------
    begin                : Mar 12, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLODESOLVERMONITOR_H_
#define TNLODESOLVERMONITOR_H_

#include <solvers/tnlIterativeSolverMonitor.h>

template< typename Real, typename Index>
class tnlODESolverMonitor : public tnlIterativeSolverMonitor< Real, Index >
{
   public:

   typedef Real RealType;
   typedef Index IndexType;

   tnlODESolverMonitor();

   void refresh();

   void setTimeStep( const RealType& timeStep );

   const RealType& getTimeStep() const;

   void setTime( const RealType& time );

   const RealType& getTime() const;

   protected:

   RealType timeStep;

   RealType time;

};

#include <solvers/ode/tnlODESolverMonitor_impl.h>

#endif /* TNLODESOLVERMONITOR_H_ */
