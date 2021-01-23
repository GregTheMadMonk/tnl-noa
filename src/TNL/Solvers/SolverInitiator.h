/***************************************************************************
                          SolverInitiator.h  -  description
                             -------------------
    begin                : Feb 23, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Solvers/BuildConfigTags.h>

namespace TNL {
namespace Solvers {

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter, typename CommunicatorType > class ProblemSetter,
          typename ConfigTag >
class SolverInitiator
{
   public:

   static bool run( const Config::ParameterContainer& parameters );

};

} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/SolverInitiator_impl.h>
