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
#include <TNL/Solvers/SolverInitiator.h>
#include <TNL/Solvers/SolverStarter.h>
#include <TNL/Solvers/SolverConfig.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/MPI/ScopedInitializer.h>
#include <TNL/MPI/Config.h>

namespace TNL {
namespace Solvers {

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          template< typename ConfigTag > class ProblemConfig,
          typename ConfigTag = DefaultBuildConfigTag >
struct Solver
{
   static bool run( int argc, char* argv[] )
   {
      Config::ParameterContainer parameters;
      Config::ConfigDescription configDescription;
      ProblemConfig< ConfigTag >::configSetup( configDescription );
      SolverConfig< ConfigTag, ProblemConfig< ConfigTag> >::configSetup( configDescription );
      configDescription.addDelimiter( "Parallelization setup:" );
      Devices::Host::configSetup( configDescription );
      Devices::Cuda::configSetup( configDescription );
      MPI::configSetup( configDescription );

      TNL::MPI::ScopedInitializer mpi( argc, argv );

      if( ! parseCommandLine( argc, argv, configDescription, parameters ) )
         return false;

      SolverInitiator< ProblemSetter, ConfigTag > solverInitiator;
      return solverInitiator.run( parameters );
   }
};

} // namespace Solvers
} // namespace TNL
