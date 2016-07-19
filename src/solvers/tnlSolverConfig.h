/***************************************************************************
                          tnlSolverConfig.h  -  description
                             -------------------
    begin                : Jul 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <config/tnlConfigDescription.h>

namespace TNL {

template< typename MeshConfig,
          typename ProblemConfig >
class tnlSolverConfig
{
   public:
      static bool configSetup( tnlConfigDescription& configDescription );
};

} // namespace TNL

#include <solvers/tnlSolverConfig_impl.h>
