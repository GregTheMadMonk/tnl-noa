/***************************************************************************
                          Solver_impl.h  -  description
                             -------------------
    begin                : Mar 9, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Solvers/SolverInitiator.h>
#include <TNL/Solvers/SolverStarter.h>
#include <TNL/Solvers/SolverConfig.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/MPI/ScopedInitializer.h>

namespace TNL {
namespace Solvers {

template< template< typename Real, typename Device, typename Index, typename MeshType, typename MeshConfig, typename SolverStarter, typename CommunicatorType > class ProblemSetter,
          template< typename MeshConfig > class ProblemConfig,
          typename MeshConfig >
bool
Solver< ProblemSetter, ProblemConfig, MeshConfig >::
run( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription configDescription;
   ProblemConfig< MeshConfig >::configSetup( configDescription );
   SolverConfig< MeshConfig, ProblemConfig< MeshConfig> >::configSetup( configDescription );
   configDescription.addDelimiter( "Parallelization setup:" );
   Devices::Host::configSetup( configDescription );
   Devices::Cuda::configSetup( configDescription );
   Communicators::MpiCommunicator::configSetup( configDescription );

   TNL::MPI::ScopedInitializer mpi( argc, argv );

   if( ! parseCommandLine( argc, argv, configDescription, parameters ) )
      return false;

   SolverInitiator< ProblemSetter, MeshConfig > solverInitiator;
   return solverInitiator.run( parameters );
};

} // namespace Solvers
} // namespace TNL
