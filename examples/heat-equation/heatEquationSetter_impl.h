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

template< typename MeshType, typename SolverStarter >
template< typename RealType, typename DeviceType, typename IndexType, typename TimeFunction>
bool heatEquationSetter< MeshType, SolverStarter > ::setAnalyticSpaceFunction (const tnlParameterContainer& parameters)
{
   SolverStarter solverStarter;
   
   const tnlString& analyticSpaceFunctionParameter = parameters.GetParameter<tnlString>("analytic-space-function");
   
   if (analyticSpaceFunctionParameter == "sin-wave")
      return solverStarter.run< heatEquationSolver< MeshType,tnlLinearDiffusion<MeshType>,
                                 tnlDirichletBoundaryConditions<MeshType>,tnlRightHandSide<MeshType>,
                                 TimeFunction,tnlSinWaveFunction<MeshType::Dimensions,Vertex,DeviceType>>>(parameters);
   if (analyticSpaceFunctionParameter == "sin-bumps")
     return solverStarter.run< heatEquationSolver< MeshType,tnlLinearDiffusion<MeshType>,
                               tnlDirichletBoundaryConditions<MeshType>,tnlRightHandSide<MeshType>,
                               TimeFunction, tnlSinBumpsFunction<MeshType::Dimensions,Vertex,DeviceType>>>(parameters);
   if (analyticSpaceFunctionParameter == "exp-bump")
      return solverStarter.run< heatEquationSolver< MeshType,tnlLinearDiffusion<MeshType>,
                                tnlDirichletBoundaryConditions<MeshType>,tnlRightHandSide<MeshType>,
                                TimeFunction, tnlExpBumpFunction<MeshType::Dimensions,Vertex,DeviceType>>>(parameters);
   
   cerr<<"Unknown analytic-space-function parameter: "<<analyticSpaceFunctionParameter<<". ";
   return 0;
}

template< typename MeshType, typename SolverStarter >
template< typename RealType, typename DeviceType, typename IndexType>
bool heatEquationSetter< MeshType, SolverStarter > ::setTimeFunction (const tnlParameterContainer& parameters)
{
   const tnlString& timeFunctionParameter = parameters.GetParameter<tnlString>("time-function");
   
   if (timeFunctionParameter == "time-independent")
      return setAnalyticSpaceFunction<RealType, DeviceType, IndexType, TimeIndependent>(parameters);
   if (timeFunctionParameter == "linear")
      return setAnalyticSpaceFunction<RealType, DeviceType, IndexType, Linear>(parameters);
   if (timeFunctionParameter == "quadratic")
      return setAnalyticSpaceFunction<RealType, DeviceType, IndexType, Quadratic>(parameters);
   if (timeFunctionParameter == "cosinus")
      return setAnalyticSpaceFunction<RealType, DeviceType, IndexType, Cosinus>(parameters);
   
   cerr<<"Unknown time-function parameter: "<<timeFunctionParameter<<". ";
   return 0;
}


template< typename MeshType,
          typename SolverStarter >
   template< typename RealType,
             typename DeviceType,
             typename IndexType >
bool heatEquationSetter< MeshType, SolverStarter > :: run( const tnlParameterContainer& parameters )
{
   return setTimeFunction<RealType, DeviceType, IndexType>(parameters);
}


#endif /* HEATEQUATIONSETTER_IMPL_H_ */
