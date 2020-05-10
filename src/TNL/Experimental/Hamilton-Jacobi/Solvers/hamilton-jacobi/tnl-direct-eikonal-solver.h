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

#include <TNL/Solvers/Solver.h>
#include <TNL/Solvers/FastBuildConfigTag.h>
#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Functions/Analytic/Constant.h>
#include <TNL/Functions/MeshFunctionView.h>
#include <TNL/Meshes/Grid.h>
#include "tnlDirectEikonalProblem.h"
#include "MainBuildConfig.h"

using namespace TNL;

//typedef tnlDefaultBuildMeshConfig BuildConfig;
//typedef Solvers::FastBuildConfig BuildConfig;
typedef HamiltonJacobiBuildConfig BuildConfig;

template< typename MeshConfig >
class DirectEikonalSolverConfig
{
   public:
      static void configSetup( Config::ConfigDescription& config )
      {
         config.addDelimiter( "Direct eikonal equation solver settings:" );
         config.addRequiredEntry< String >( "input-file", "Input file." );
         config.addEntry< String >( "distributed-grid-io-type", "Choose Distributed Grid IO Type", "MpiIO");
            config.addEntryEnum< String >( "LocalCopy" );
            config.addEntryEnum< String >( "MpiIO" );
      };
};

template< typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename MeshConfig,
          typename SolverStarter,
          typename CommunicatorType >
class DirectEikonalSolverSetter
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef Containers::StaticVector< MeshType::getMeshDimension(), Real > Point;

   static bool run( const Config::ParameterContainer& parameters )
   {
      static const int Dimension = MeshType::getMeshDimension();
      typedef Functions::Analytic::Constant< Dimension, Real > Anisotropy;
      typedef tnlDirectEikonalProblem< MeshType, CommunicatorType, Anisotropy > Problem;
      SolverStarter solverStarter;
      return solverStarter.template run< Problem >( parameters );
   };
};

int main( int argc, char* argv[] )
{
   if( ! Solvers::Solver< DirectEikonalSolverSetter, DirectEikonalSolverConfig, BuildConfig >::run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}


