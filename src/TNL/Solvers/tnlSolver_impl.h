/***************************************************************************
                          tnlSolver_impl.h  -  description
                             -------------------
    begin                : Mar 9, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Solvers/tnlSolverInitiator.h>
#include <TNL/Solvers/tnlSolverStarter.h>
#include <TNL/Solvers/tnlSolverConfig.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Solvers {
   
template< template< typename Real, typename Device, typename Index, typename MeshType, typename MeshConfig, typename SolverStarter > class ProblemSetter,
          template< typename MeshConfig > class ProblemConfig,
          typename MeshConfig >
bool
tnlSolver< ProblemSetter, ProblemConfig, MeshConfig >::
run( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription configDescription;
   ProblemConfig< MeshConfig >::configSetup( configDescription );
   tnlSolverConfig< MeshConfig, ProblemConfig< MeshConfig> >::configSetup( configDescription );
   configDescription.addDelimiter( "Parallelization setup:" );
   Devices::Host::configSetup( configDescription );
   Devices::Cuda::configSetup( configDescription );

   if( ! parseCommandLine( argc, argv, configDescription, parameters ) )
      return false;

   tnlSolverInitiator< ProblemSetter, MeshConfig > solverInitiator;
   return solverInitiator.run( parameters );
};

} // namespace Solvers
} // namespace TNL
