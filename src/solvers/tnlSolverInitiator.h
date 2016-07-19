/***************************************************************************
                          tnlSolverInitiator.h  -  description
                             -------------------
    begin                : Feb 23, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <tnlObject.h>
#include <config/tnlParameterContainer.h>
#include <solvers/tnlBuildConfigTags.h>

namespace TNL {

template< template< typename Real, typename Device, typename Index, typename MeshType, typename MeshConfig, typename SolverStarter > class ProblemSetter,
          typename MeshConfig >
class tnlSolverInitiator : public tnlObject
{
   public:

   static bool run( const tnlParameterContainer& parameters );

};

} // namespace TNL

#include <solvers/tnlSolverInitiator_impl.h>
