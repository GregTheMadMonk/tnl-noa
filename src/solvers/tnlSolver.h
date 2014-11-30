/***************************************************************************
                          tnlSolver.h  -  description
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

#ifndef TNLSOLVER_H_
#define TNLSOLVER_H_

#include <solvers/tnlConfigTags.h>

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          template< typename ConfTag > class ProblemConfig,
          typename ConfigTag = tnlDefaultConfigTag >
class tnlSolver
{
   public:
   bool run( int argc, char* argv[] );

   protected:
};

#include <implementation/solvers/tnlSolver_impl.h>
#endif /* TNLSOLVER_H_ */
