/***************************************************************************
                          hamiltonJacobiProblemSetter_impl.h  -  description
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

#include <mesh/tnlGrid.h>
#include <functions/tnlConstantFunction.h>
#include <operators/tnlNeumannBoundaryConditions.h>
#include <operators/tnlDirichletBoundaryConditions.h>
#include <operators/hamilton-jacobi/upwindEikonal.h>
#include <operators/hamilton-jacobi/godunovEikonal.h>
#include <operators/hamilton-jacobi/tnlEikonalOperator.h>


template< typename RealType,
          typename DeviceType,
          typename IndexType,
          typename MeshType,
          typename ConfigTag,
          typename SolverStarter >
bool HamiltonJacobiProblemSetter< RealType, DeviceType, IndexType, MeshType, ConfigTag, SolverStarter > :: run( const tnlParameterContainer& parameters )
{
   static const int Dimensions = MeshType::getMeshDimensions();

   if( Dimensions <= 0 || Dimensions > 3 )
   {
      cerr << "The problem is not defined for " << Dimensions << "dimensions." << endl;
      return false;
   }
   else
   {
      typedef tnlStaticVector < Dimensions, RealType > Vertex;
      typedef tnlConstantFunction< Dimensions, RealType > ConstantFunctionType;
      typedef tnlNeumannBoundaryConditions< MeshType, ConstantFunctionType, RealType, IndexType > BoundaryConditions;

      SolverStarter solverStarter;

      const tnlString& schemeName = parameters.getParameter< tnlString >( "scheme" );

      if( schemeName == "upwind" )
      {
           typedef upwindEikonalScheme< MeshType, RealType, IndexType > GradientNormOperator;
           typedef tnlConstantFunction< Dimensions, RealType > RightHandSide;
           typedef tnlEikonalOperator< GradientNormOperator, RightHandSide > Operator;
           typedef HamiltonJacobiProblem< MeshType, Operator, BoundaryConditions, RightHandSide > Solver;
           return solverStarter.template run< Solver >( parameters );
      }
      if( schemeName == "godunov" )
      {
           typedef godunovEikonalScheme< MeshType, RealType, IndexType > GradientNormOperator;
           typedef tnlConstantFunction< Dimensions, RealType > RightHandSide;
           typedef tnlEikonalOperator< GradientNormOperator, RightHandSide > Operator;
           typedef HamiltonJacobiProblem< MeshType, Operator, BoundaryConditions, RightHandSide > Solver;
           return solverStarter.template run< Solver >( parameters );
      }      
      else
         cerr << "Unknown scheme '" << schemeName << "'." << endl;


      return false;
   }
}
