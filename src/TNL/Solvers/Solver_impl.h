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

#include <TNL/mpi-supp.h>

namespace TNL {
namespace Solvers {
   
template< template< typename Real, typename Device, typename Index, typename MeshType, typename MeshConfig, typename SolverStarter > class ProblemSetter,
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

#ifdef USE_MPI
	MPI::Init(argc,argv);
    //redirect all stdout to files, only 0 take to go to console
    std::streambuf *psbuf, *backup;
    std::ofstream filestr;
    backup=std::cout.rdbuf();

    //redirect output to files...
    if(MPI::COMM_WORLD.Get_rank()!=0)
    {
        String stdoutfile;
        stdoutfile=String( "./stdout-")+convertToString(MPI::COMM_WORLD.Get_rank())+String(".txt");
        filestr.open (stdoutfile.getString()); 
        psbuf = filestr.rdbuf(); 
        std::cout.rdbuf(psbuf);
    }
#endif

   if( ! parseCommandLine( argc, argv, configDescription, parameters ) )
      return false;

   SolverInitiator< ProblemSetter, MeshConfig > solverInitiator;
   bool ret= solverInitiator.run( parameters );

#ifdef USE_MPI
    //redirect output to files...
    if(MPI::COMM_WORLD.Get_rank()!=0)
    { 
        std::cout.rdbuf(backup);
        filestr.close(); 
    }
	MPI::Finalize();
#endif
	return ret;
};

} // namespace Solvers
} // namespace TNL
