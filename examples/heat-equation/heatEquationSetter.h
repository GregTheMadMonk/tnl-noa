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
#include <functions/tnlSinWaveFunction.h>
#include <functions/tnlExpBumpFunction.h>
#include <functions/tnlSinBumpsFunction.h>
#include "tnlTimeFunction.h"
#include "tnlDirichletBoundaryConditions.h"
#include <operators/diffusion/tnlLinearDiffusion.h>
#include "tnlNeumannBoundaryConditions.h"
#include "tnlZeroRightHandSide.h"
#include "tnlRightHandSide.h"

   
template< typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag,
          typename SolverStarter >
class heatEquationSetter
{
   public:
 
   typedef TimeFunction< MeshType, TimeFunctionBase::TimeIndependent, Real, Index > TimeIndependent;
   typedef TimeFunction< MeshType, TimeFunctionBase::Linear, Real, Index > Linear;
   typedef TimeFunction< MeshType, TimeFunctionBase::Quadratic, Real, Index > Quadratic;
   typedef TimeFunction< MeshType, TimeFunctionBase::Cosinus, Real, Index > Cosinus;
   //typedef typename MeshType::RealType RealType;

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef tnlStaticVector< MeshType::Dimensions, Real > Vertex;
      
   template< typename TimeFunction >
   static bool setAnalyticSpaceFunction (const tnlParameterContainer& parameters);  
    
   static bool setTimeFunction (const tnlParameterContainer& parameters);
      
   static bool run( const tnlParameterContainer& parameters );
};

#include "heatEquationSetter_impl.h"

#endif /* HEATEQUATIONSETTER_H_ */
