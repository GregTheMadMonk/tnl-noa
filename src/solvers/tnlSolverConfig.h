/***************************************************************************
                          tnlSolverConfig.h  -  description
                             -------------------
    begin                : Jul 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLSOLVERCONFIG_H_
#define TNLSOLVERCONFIG_H_

#include <config/tnlConfigDescription.h>

template< typename ConfigTag,
          typename ProblemConfig >
class tnlSolverConfig
{
   public:
      static bool configSetup( tnlConfigDescription& configDescription );
};

#include <implementation/solvers/tnlSolverConfig_impl.h>

#endif /* TNLSOLVERCONFIG_H_ */
