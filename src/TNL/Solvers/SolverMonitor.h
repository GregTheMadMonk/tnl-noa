/***************************************************************************
                          SolverMonitor.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Solvers {   

template< typename Real, typename Index >
class SolverMonitor
{
   public:

   virtual void refresh( bool force = false ) = 0;

   ~SolverMonitor() {};
 
};

} // namespace Solvers
} // namespace TNL

