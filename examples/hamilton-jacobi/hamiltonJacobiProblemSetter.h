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

#ifndef HAMILTONJACOBIPROBLEMSETTER_H_
#define HAMILTONJACOBIPROBLEMSETTER_H_

#include <config/tnlParameterContainer.h>
#include <mesh/tnlGrid.h>
#include <functions/tnlConstantFunction.h>
#include <operators/tnlNeumannReflectionBoundaryConditions.h>
#include <operators/tnlDirichletBoundaryConditions.h>
#include "hamiltonJacobiProblemSolver.h"
#include <operators/upwind-eikonal/upwindEikonal.h>
#include <operators/godunov-eikonal/godunovEikonal.h>
#include <operators/upwind/upwind.h>
#include <operators/godunov/godunov.h>
#include <functions/tnlSDFSign.h>
#include <functions/tnlSDFGridValue.h>

template< typename RealType,
		  typename DeviceType,
		  typename IndexType,
		  typename MeshType,
		  typename ConfigTag,
          typename SolverStarter >
class hamiltonJacobiProblemSetter
{
   public:
   static bool run( const tnlParameterContainer& parameters );
};

#include "hamiltonJacobiProblemSetter_impl.h"

#endif /* HAMILTONJACOBIPROBLEMSETTER_H_ */
