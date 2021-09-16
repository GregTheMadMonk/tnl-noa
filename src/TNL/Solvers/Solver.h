/***************************************************************************
                          Solver.h  -  description
                             -------------------
    begin                : Mar 9, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Solvers/BuildConfigTags.h>

namespace TNL {
namespace Solvers {

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          template< typename ConfTag > class ProblemConfig,
          typename ConfigTag = DefaultBuildConfigTag >
class Solver
{
   public:
   static bool run( int argc, char* argv[] );

   protected:
};

} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Solver_impl.h>
