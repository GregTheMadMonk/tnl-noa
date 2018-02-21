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
#include <TNL/Devices/Cuda.h>
#include <TNL/Communicators/NoDistrCommunicator.h>
#include <TNL/Communicators/MpiCommunicator.h>

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

    //iniicialization needs argc and argc-> needs to be close to main
       Communicators::NoDistrCommunicator::Init(argc,argv, true);
#ifdef HAVE_MPI
       Communicators::MpiCommunicator::Init(argc,argv, true);
#endif

   if( ! parseCommandLine( argc, argv, configDescription, parameters ) )
      return false;

   SolverInitiator< ProblemSetter, MeshConfig > solverInitiator;
   bool ret= solverInitiator.run( parameters );

	return ret;
};

} // namespace Solvers
} // namespace TNL
