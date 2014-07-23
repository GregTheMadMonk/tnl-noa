/***************************************************************************
                          simpleProblemSetter_impl.h  -  description
                             -------------------
    begin                : Mar 10, 2013
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

#ifndef HEATEQUATIONSETTER_IMPL_H_
#define HEATEQUATIONSETTER_IMPL_H_

#include "heatEquationSetter.h"
#include "heatEquationSolver.h"
#include <generators/functions/tnlSinWaveFunction.h>
#include <generators/functions/tnlSinBumpsFunction.h>
#include <generators/functions/tnlExpBumpFunction.h>
#include "tnlLinearDiffusion.h"
#include "tnlDirichletBoundaryConditions.h"
#include "tnlRightHandSide.h"

template< typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag,
          typename SolverStarter >
   template< typename TimeFunction >          
bool heatEquationSetter< Real, Device, Index, MeshType, ConfigTag, SolverStarter > ::setAnalyticSpaceFunction (const tnlParameterContainer& parameters)
{
   SolverStarter solverStarter;
   
   //DODELAT NACTENI Z PRIKAZOVY RADKY: RHS, Diffusion, BoundaryConditions !!!!!
  
   const tnlString& analyticSpaceFunctionParameter = parameters.GetParameter<tnlString>("test-function");
   
   typedef tnlLinearDiffusion< MeshType, Real, Index > Scheme;
   typedef tnlDirichletBoundaryConditions< MeshType, Real, Index > BoundaryConditions;
   typedef tnlRightHandSide< MeshType, Real, Index > RightHandSide;
   if (analyticSpaceFunctionParameter == "sin-wave")
   {
      typedef tnlSinWaveFunction< MeshType::Dimensions, Vertex, DeviceType > TestFunction;
      typedef heatEquationSolver< MeshType, Scheme, BoundaryConditions, RightHandSide, TimeFunction, TestFunction > Solver;
      return solverStarter.template run< Solver >( parameters );
   }
   if (analyticSpaceFunctionParameter == "sin-bumps")
   {
      typedef tnlSinBumpsFunction<MeshType::Dimensions,Vertex,DeviceType > TestFunction;
      typedef heatEquationSolver< MeshType, Scheme, BoundaryConditions, RightHandSide, TimeFunction, TestFunction > Solver;
      return solverStarter.template run< Solver >( parameters );
   }
   if (analyticSpaceFunctionParameter == "exp-bump")
   {
      typedef tnlExpBumpFunction<MeshType::Dimensions,Vertex,DeviceType > TestFunction;
      typedef heatEquationSolver< MeshType, Scheme, BoundaryConditions, RightHandSide, TimeFunction, TestFunction > Solver;
      return solverStarter.template run< Solver >( parameters );
   }
   
   cerr<<"Unknown test-function parameter: "<<analyticSpaceFunctionParameter<<". ";
   return 0;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag,
          typename SolverStarter >
bool heatEquationSetter< Real, Device, Index, MeshType, ConfigTag, SolverStarter > ::setTimeFunction (const tnlParameterContainer& parameters)
{
   const tnlString& timeFunctionParameter = parameters.GetParameter<tnlString>("test-function-time-dependence");
   
   if (timeFunctionParameter == "none")
      return setAnalyticSpaceFunction< TimeIndependent >(parameters);
   if (timeFunctionParameter == "linear")
      return setAnalyticSpaceFunction< Linear >(parameters);
   if (timeFunctionParameter == "quadratic")
      return setAnalyticSpaceFunction< Quadratic >(parameters);
   if (timeFunctionParameter == "cosine")
      return setAnalyticSpaceFunction< Cosinus >(parameters);
   
   cerr<<"Unknown time-function parameter: "<<timeFunctionParameter<<". ";
   return 0;
}


template< typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag,
          typename SolverStarter >
bool heatEquationSetter< Real, Device, Index, MeshType, ConfigTag, SolverStarter >::run( const tnlParameterContainer& parameters )
{
   return setTimeFunction(parameters);
}


#endif /* HEATEQUATIONSETTER_IMPL_H_ */
