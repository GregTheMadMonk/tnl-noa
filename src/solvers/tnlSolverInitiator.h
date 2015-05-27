/***************************************************************************
                          tnlSolverInitiator.h  -  description
                             -------------------
    begin                : Feb 23, 2013
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

#ifndef TNLSOLVERINITIATOR_H_
#define TNLSOLVERINITIATOR_H_

#include <core/tnlObject.h>
#include <config/tnlParameterContainer.h>
#include <solvers/tnlBuildConfigTags.h>

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename ConfigTag >
class tnlSolverInitiator : public tnlObject
{
   public:

   static bool run( const tnlParameterContainer& parameters );

};

#include <solvers/tnlSolverInitiator_impl.h>

#endif /* TNLSOLVERINITIATOR_H_ */
