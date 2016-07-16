/***************************************************************************
                          tnlSolver.h  -  description
                             -------------------
    begin                : Mar 9, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLSOLVER_H_
#define TNLSOLVER_H_

#include <solvers/tnlBuildConfigTags.h>

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          template< typename ConfTag > class ProblemConfig,
          typename ConfigTag = tnlDefaultBuildConfigTag >
class tnlSolver
{
   public:
   static bool run( int argc, char* argv[] );

   protected:
};

#include <solvers/tnlSolver_impl.h>
#endif /* TNLSOLVER_H_ */
