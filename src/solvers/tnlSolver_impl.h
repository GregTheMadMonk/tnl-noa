/***************************************************************************
                          tnlSolver_impl.h  -  description
                             -------------------
    begin                : Mar 9, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLSOLVER_IMPL_H_
#define TNLSOLVER_IMPL_H_

#include <solvers/tnlSolverInitiator.h>
#include <solvers/tnlSolverStarter.h>
#include <solvers/tnlSolverConfig.h>
#include <core/tnlOmp.h>
#include <core/tnlCuda.h>

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
   tnlOmp::configSetup( configDescription );
   tnlCuda::configSetup( configDescription );

   if( ! parseCommandLine( argc, argv, configDescription, parameters ) )
      return false;

   tnlSolverInitiator< ProblemSetter, MeshConfig > solverInitiator;
   return solverInitiator.run( parameters );
};

#endif /* TNLSOLVER_IMPL_H_ */
