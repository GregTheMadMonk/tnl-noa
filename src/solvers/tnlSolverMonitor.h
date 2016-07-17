/***************************************************************************
                          tnlSolverMonitor.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLSOLVERMONITOR_H_
#define TNLSOLVERMONITOR_H_

template< typename Real, typename Index >
class tnlSolverMonitor
{
   public:

   virtual void refresh( bool force = false ) = 0;

   ~tnlSolverMonitor() {};
 
};


#endif /* TNLSOLVERMONITOR_H_ */
