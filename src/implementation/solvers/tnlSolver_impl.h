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

template< template< typename MeshType, typename SolverStarter > class ProblemSetter,
          typename SolverConfig >
bool tnlSolver< ProblemSetter, SolverConfig > :: run( const char* configFileName, int argc, char* argv[] )
{
   tnlSolverInitiator< ProblemSetter, SolverConfig > solverInitiator;
   if( ! solverInitiator. run( configFileName, argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;

};

#endif /* TNLSOLVER_IMPL_H_ */
