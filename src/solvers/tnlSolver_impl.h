/***************************************************************************
                          tnlSolver_impl.h  -  description
                             -------------------
    begin                : Mar 9, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <solvers/tnlSolverInitiator.h>
#include <solvers/tnlSolverStarter.h>
#include <solvers/tnlSolverConfig.h>
#include <core/tnlCuda.h>

namespace TNL {

template< template< typename Real, typename Device, typename Index, typename MeshType, typename MeshConfig, typename SolverStarter > class ProblemSetter,
          template< typename MeshConfig > class ProblemConfig,
          typename MeshConfig >
bool
tnlSolver< ProblemSetter, ProblemConfig, MeshConfig >::
run( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription configDescription;
   ProblemConfig< MeshConfig >::configSetup( configDescription );
   tnlSolverConfig< MeshConfig, ProblemConfig< MeshConfig> >::configSetup( configDescription );
   configDescription.addDelimiter( "Parallelization setup:" );
   tnlHost::configSetup( configDescription );
   tnlCuda::configSetup( configDescription );

   if( ! parseCommandLine( argc, argv, configDescription, parameters ) )
      return false;

   tnlSolverInitiator< ProblemSetter, MeshConfig > solverInitiator;
   return solverInitiator.run( parameters );
};

} // namespace TNL
