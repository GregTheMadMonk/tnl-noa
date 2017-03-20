/***************************************************************************
                          hamiltonJacobiProblemSetter.h  -  description
                             -------------------
    begin                : Jul 8 , 2014
    copyright            : (C) 2014 by Tomas Sobotik
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#pragma once 

#include <TNL/Config/ParameterContainer.h>
#include "HamiltonJacobiProblem.h"

template< typename RealType,
		  typename DeviceType,
		  typename IndexType,
		  typename MeshType,
		  typename ConfigTag,
          typename SolverStarter >
class HamiltonJacobiProblemSetter
{
   public:
   static bool run( const Config::ParameterContainer& parameters );
};

#include "HamiltonJacobiProblemSetter_impl.h"

