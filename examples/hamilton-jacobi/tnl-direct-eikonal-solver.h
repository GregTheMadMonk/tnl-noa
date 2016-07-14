/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   tnl-direct-eikonal-solver.h
 * Author: oberhuber
 *
 * Created on July 13, 2016, 1:09 PM
 */

#pragma once

#include <solvers/tnlSolver.h>
#include <solvers/tnlFastBuildConfigTag.h>
#include <solvers/tnlBuildConfigTags.h>
#include <functions/tnlConstantFunction.h>
#include <functions/tnlMeshFunction.h>
//#include <problems/tnlHeatEquationProblem.h>
#include <mesh/tnlGrid.h>
#include "tnlDirectEikonalProblem.h"

//typedef tnlDefaultBuildMeshConfig BuildConfig;
typedef tnlFastBuildConfig BuildConfig;

template< typename MeshConfig >
class tnlDirectEikonalSolverConfig
{
   public:
      static void configSetup( tnlConfigDescription& config )
      {
         config.addDelimiter( "Direct eikonal equation solver settings:" );
         config.addRequiredEntry< tnlString >( "input-file", "Input file." );
      };
};

template< typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename MeshConfig,
          typename SolverStarter >
class tnlDirectEikonalSolverSetter
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef tnlStaticVector< MeshType::meshDimensions, Real > Vertex;

   static bool run( const tnlParameterContainer& parameters )
   {
      enum { Dimensions = MeshType::meshDimensions };
      typedef tnlConstantFunction< Dimensions, Real > Anisotropy;
      typedef tnlDirectEikonalProblem< MeshType, Anisotropy > Problem;
      SolverStarter solverStarter;
      return solverStarter.template run< Problem >( parameters );
   };
};

int main( int argc, char* argv[] )
{
   if( ! tnlSolver< tnlDirectEikonalSolverSetter, tnlDirectEikonalSolverConfig, BuildConfig >::run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}


