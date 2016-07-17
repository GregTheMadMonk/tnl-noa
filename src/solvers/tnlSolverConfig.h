/***************************************************************************
                          tnlSolverConfig.h  -  description
                             -------------------
    begin                : Jul 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLSOLVERCONFIG_H_
#define TNLSOLVERCONFIG_H_

#include <config/tnlConfigDescription.h>

template< typename MeshConfig,
          typename ProblemConfig >
class tnlSolverConfig
{
   public:
      static bool configSetup( tnlConfigDescription& configDescription );
};

#include <solvers/tnlSolverConfig_impl.h>

#endif /* TNLSOLVERCONFIG_H_ */
