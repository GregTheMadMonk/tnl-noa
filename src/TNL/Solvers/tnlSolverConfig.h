/***************************************************************************
                          tnlSolverConfig.h  -  description
                             -------------------
    begin                : Jul 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ConfigDescription.h>

namespace TNL {

template< typename MeshConfig,
          typename ProblemConfig >
class tnlSolverConfig
{
   public:
      static bool configSetup( Config::ConfigDescription& configDescription );
};

} // namespace TNL

#include <TNL/Solvers/tnlSolverConfig_impl.h>
