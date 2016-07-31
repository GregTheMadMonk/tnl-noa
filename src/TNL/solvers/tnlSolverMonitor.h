/***************************************************************************
                          tnlSolverMonitor.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

template< typename Real, typename Index >
class tnlSolverMonitor
{
   public:

   virtual void refresh( bool force = false ) = 0;

   ~tnlSolverMonitor() {};
 
};

} // namespace TNL

