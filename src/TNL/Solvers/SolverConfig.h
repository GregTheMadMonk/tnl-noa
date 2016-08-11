/***************************************************************************
                          SolverConfig.h  -  description
                             -------------------
    begin                : Jul 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ConfigDescription.h>

namespace TNL {
namespace Solvers {   

template< typename MeshConfig,
          typename ProblemConfig >
class SolverConfig
{
   public:
      static bool configSetup( Config::ConfigDescription& configDescription );
};

} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/SolverConfig_impl.h>
