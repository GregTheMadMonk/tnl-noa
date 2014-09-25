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

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          template< typename ConfTag > class ProblemConfig,
          typename ConfigTag >
bool tnlSolver< ProblemSetter, ProblemConfig, ConfigTag > :: run( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription configDescription;
   ProblemConfig< ConfigTag >::configSetup( configDescription );
   tnlSolverConfig< ConfigTag, ProblemConfig< ConfigTag> >::configSetup( configDescription );
   if( ! ParseCommandLine( argc, argv, configDescription, parameters ) )
      return false;

   tnlSolverInitiator< ProblemSetter, ConfigTag > solverInitiator;
   if( ! solverInitiator. run( parameters ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;

};

#endif /* TNLSOLVER_IMPL_H_ */
