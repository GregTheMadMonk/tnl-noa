/***************************************************************************
                          simpleProblemSetter.h  -  description
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

#ifndef SIMPLEPROBLEMTYPESSETTER_H_
#define SIMPLEPROBLEMTYPESSETTER_H_

#include <config/tnlParameterContainer.h>
#include <mesh/tnlGrid.h>
#include "heatEquationSolver.h"
#include <generators/functions/tnlSinWaveFunction.h>
#include <generators/functions/tnlExpBumpFunction.h>
#include <generators/functions/tnlSinBumpsFunction.h>
#include "tnlTimeFunction.h"
#include "tnlDirichletBoundaryConditions.h"
#include "tnlLinearDiffusion.h"
#include "tnlNeumannBoundaryConditions.h"
#include "tnlZeroRightHandSide.h"
#include "tnlRightHandSide.h"

   
template< typename MeshType,
          typename SolverStarter >
class heatEquationSetter
{
   public:
 
   typedef TimeFunction<MeshType,TimeFunctionBase::TimeIndependent> TimeIndependent;
   typedef TimeFunction<MeshType,TimeFunctionBase::Linear> Linear;
   typedef TimeFunction<MeshType,TimeFunctionBase::Quadratic> Quadratic;
   typedef TimeFunction<MeshType,TimeFunctionBase::Cosinus> Cosinus;
   typedef typename MeshType::RealType RealType;
   typedef tnlStaticVector<MeshType::Dimensions, RealType> Vertex;

      
   template< typename RealType, typename DeviceType, typename IndexType, typename TimeFunction>
   static bool setAnalyticSpaceFunction (const tnlParameterContainer& parameters);  
    
   template< typename RealType, typename DeviceType, typename IndexType>
   static bool setTimeFunction (const tnlParameterContainer& parameters);
      
   template< typename RealType,
             typename DeviceType,
             typename IndexType >
   static bool run( const tnlParameterContainer& parameters );
};

#include "heatEquationSetter_impl.h"

#endif /* HEATEQUATIONSETTER_H_ */
