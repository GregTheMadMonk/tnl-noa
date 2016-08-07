/***************************************************************************
                          tnlSolverInitiator.h  -  description
                             -------------------
    begin                : Feb 23, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Object.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Solvers/tnlBuildConfigTags.h>

namespace TNL {

template< template< typename Real, typename Device, typename Index, typename MeshType, typename MeshConfig, typename SolverStarter > class ProblemSetter,
          typename MeshConfig >
class tnlSolverInitiator : public Object
{
   public:

   static bool run( const Config::ParameterContainer& parameters );

};

} // namespace TNL

#include <TNL/Solvers/tnlSolverInitiator_impl.h>
