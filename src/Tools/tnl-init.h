/***************************************************************************
                          tnl-init.h  -  description
                             -------------------
    begin                : Nov 23, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/TestFunction.h>
#include <TNL/Operators/FiniteDifferences.h>
#include <TNL/FileName.h>
#include <TNL/Functions/MeshFunction.h>

#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Meshes/DistributedMeshes/DistributedGridIO.h>
#include <TNL/Meshes/DistributedMeshes/SubdomainOverlapsGetter.h>

#include <TNL/Communicators/MpiCommunicator.h>

using namespace TNL;

template< typename MeshType,
          typename RealType,
          int xDiff,
          int yDiff,
          int zDiff >
bool renderFunction( const Config::ParameterContainer& parameters )
{
   using CommunicatorType = Communicators::MpiCommunicator;

   using namespace  Meshes::DistributedMeshes;
   using DistributedGridType = Meshes::DistributedMeshes::DistributedMesh<MeshType>;
   DistributedGridType distributedMesh;
   Pointers::SharedPointer< MeshType > meshPointer;
   MeshType globalMesh;

   if(CommunicatorType::isDistributed())
   {
       //suppose global mesh loaded from single file
       String meshFile = parameters.getParameter< String >( "mesh" );
       std::cout << "+ -> Loading mesh from " << meshFile << " ... " << std::endl;
       globalMesh.load( meshFile );
   
       // TODO: This should work with no overlaps
       distributedMesh.template setGlobalGrid<CommunicatorType>(globalMesh);
       typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
       SubdomainOverlapsGetter< MeshType, CommunicatorType >::getOverlaps( &distributedMesh, lowerOverlap, upperOverlap, 1 );
       distributedMesh.setOverlaps( lowerOverlap, upperOverlap );
       distributedMesh.setupGrid(*meshPointer);
    }
    else
    {
       String meshFile = parameters.getParameter< String >( "mesh" );
       std::cout << "+ -> Loading mesh from " << meshFile << " ... " << std::endl;
       meshPointer->load( meshFile );
    }

   typedef Functions::TestFunction< MeshType::getMeshDimension(), RealType > FunctionType;
   typedef Pointers::SharedPointer<  FunctionType, typename MeshType::DeviceType > FunctionPointer;
   FunctionPointer function;
   std::cout << "Setting up the function ... " << std::endl;
   if( ! function->setup( parameters, "" ) )
      return false;
   std::cout << "done." << std::endl;
   typedef Functions::MeshFunction< MeshType, MeshType::getMeshDimension() > MeshFunctionType;
   typedef Pointers::SharedPointer<  MeshFunctionType, typename MeshType::DeviceType > MeshFunctionPointer;
   MeshFunctionPointer meshFunction( meshPointer );
   //if( ! discreteFunction.setSize( mesh.template getEntitiesCount< typename MeshType::Cell >() ) )
   //   return false;
 
   double finalTime = parameters.getParameter< double >( "final-time" );
   double initialTime = parameters.getParameter< double >( "initial-time" );
   double tau = parameters.getParameter< double >( "snapshot-period" );
   bool numericalDifferentiation = parameters.getParameter< bool >( "numerical-differentiation" );
   int step( 0 );
   double time( initialTime );
   const int steps = tau > 0 ? ceil( ( finalTime - initialTime ) / tau ): 0;

   while( step <= steps )
   {

      if( numericalDifferentiation )
      {
        std::cout << "+ -> Computing the finite differences ... " << std::endl;
         MeshFunctionType auxDiscreteFunction;
         //if( ! auxDiscreteFunction.setSize( mesh.template getEntitiesCount< typename MeshType::Cell >() ) )
         //   return false;
         //tnlFunctionDiscretizer< MeshType, FunctionType, DiscreteFunctionType >::template discretize< 0, 0, 0 >( mesh, function, auxDiscreteFunction, time );
         //FiniteDifferences< MeshType >::template getDifference< DiscreteFunctionType, xDiff, yDiff, zDiff, 0, 0, 0 >( mesh, auxDiscreteFunction, discreteFunction );
      }
      else
      {
         Functions::MeshFunctionEvaluator< MeshFunctionType, FunctionType >::evaluate( meshFunction, function, time );
      }

      String outputFile = parameters.getParameter< String >( "output-file" );
      if( finalTime > 0.0 )
      {
         String extension = getFileExtension( outputFile );
         outputFile = removeFileNameExtension( outputFile );
         outputFile += "-";
         FileName outputFileName;
         outputFileName.setFileNameBase( outputFile.getString() );
         outputFileName.setExtension( extension.getString() );
         outputFileName.setIndex( step );
         outputFile = outputFileName.getFileName();
        std::cout << "+ -> Writing the function at the time " << time << " to " << outputFile << " ... " << std::endl;
      }
      else
        std::cout << "+ -> Writing the function to " << outputFile << " ... " << std::endl;

      if(CommunicatorType::isDistributed())
      {
         if( ! Meshes::DistributedMeshes::DistributedGridIO<MeshFunctionType> ::save(outputFile, *meshFunction ) )
            return false;
      }
      else
        meshFunction->save( outputFile);

      time += tau;
      step ++;
   }
   return true;
}

template< typename MeshType,
          typename RealType >
bool resolveDerivatives( const Config::ParameterContainer& parameters )
{
   int xDiff = parameters.getParameter< int >( "x-derivative" );
   int yDiff = parameters.getParameter< int >( "y-derivative" );
   int zDiff = parameters.getParameter< int >( "z-derivative" );
   if( xDiff < 0 || yDiff < 0 || zDiff < 0 || ( xDiff + yDiff + zDiff ) > 4 )
   {
      std::cerr << "Wrong orders of partial derivatives: "
           << xDiff << " " << yDiff << " " << zDiff << ". "
           << "They can be only non-negative integer numbers in sum not larger than 4."
           << std::endl;
      return false;
   }
   if( xDiff == 0 && yDiff == 0 && zDiff == 0 )
      return renderFunction< MeshType, RealType, 0, 0, 0 >( parameters );
   if( xDiff == 0 && yDiff == 0 && zDiff == 1 )
      return renderFunction< MeshType, RealType, 0, 0, 1 >( parameters );
   if( xDiff == 0 && yDiff == 0 && zDiff == 2 )
      return renderFunction< MeshType, RealType, 0, 0, 2 >( parameters );
   if( xDiff == 0 && yDiff == 0 && zDiff == 3 )
      return renderFunction< MeshType, RealType, 0, 0, 3 >( parameters );
   if( xDiff == 0 && yDiff == 0 && zDiff == 4 )
      return renderFunction< MeshType, RealType, 0, 0, 4 >( parameters );
   if( xDiff == 0 && yDiff == 1 && zDiff == 0 )
      return renderFunction< MeshType, RealType, 0, 1, 0 >( parameters );
   if( xDiff == 0 && yDiff == 1 && zDiff == 1 )
      return renderFunction< MeshType, RealType, 0, 1, 1 >( parameters );
   if( xDiff == 0 && yDiff == 1 && zDiff == 2 )
      return renderFunction< MeshType, RealType, 0, 1, 2 >( parameters );
   if( xDiff == 0 && yDiff == 1 && zDiff == 3 )
      return renderFunction< MeshType, RealType, 0, 1, 3 >( parameters );
   if( xDiff == 0 && yDiff == 2 && zDiff == 0 )
      return renderFunction< MeshType, RealType, 0, 2, 0 >( parameters );
   if( xDiff == 0 && yDiff == 2 && zDiff == 1 )
      return renderFunction< MeshType, RealType, 0, 2, 1 >( parameters );
   if( xDiff == 0 && yDiff == 2 && zDiff == 2 )
      return renderFunction< MeshType, RealType, 0, 2, 2 >( parameters );
   if( xDiff == 0 && yDiff == 3 && zDiff == 0 )
      return renderFunction< MeshType, RealType, 0, 3, 0 >( parameters );
   if( xDiff == 0 && yDiff == 3 && zDiff == 1 )
      return renderFunction< MeshType, RealType, 0, 3, 1 >( parameters );
   if( xDiff == 0 && yDiff == 4 && zDiff == 0 )
      return renderFunction< MeshType, RealType, 0, 4, 0 >( parameters );
   if( xDiff == 1 && yDiff == 0 && zDiff == 0 )
      return renderFunction< MeshType, RealType, 1, 0, 0 >( parameters );
   if( xDiff == 1 && yDiff == 0 && zDiff == 1 )
      return renderFunction< MeshType, RealType, 1, 0, 1 >( parameters );
   if( xDiff == 1 && yDiff == 0 && zDiff == 2 )
      return renderFunction< MeshType, RealType, 1, 0, 2 >( parameters );
   if( xDiff == 1 && yDiff == 0 && zDiff == 3 )
      return renderFunction< MeshType, RealType, 1, 0, 3 >( parameters );
   if( xDiff == 1 && yDiff == 1 && zDiff == 0 )
      return renderFunction< MeshType, RealType, 1, 1, 0 >( parameters );
   if( xDiff == 1 && yDiff == 1 && zDiff == 1 )
      return renderFunction< MeshType, RealType, 1, 1, 1 >( parameters );
   if( xDiff == 1 && yDiff == 1 && zDiff == 2 )
      return renderFunction< MeshType, RealType, 1, 1, 2 >( parameters );
   if( xDiff == 1 && yDiff == 2 && zDiff == 0 )
      return renderFunction< MeshType, RealType, 1, 2, 0 >( parameters );
   if( xDiff == 1 && yDiff == 2 && zDiff == 1 )
      return renderFunction< MeshType, RealType, 1, 2, 1 >( parameters );
   if( xDiff == 1 && yDiff == 3 && zDiff == 0 )
      return renderFunction< MeshType, RealType, 1, 3, 0 >( parameters );
   if( xDiff == 2 && yDiff == 0 && zDiff == 0 )
      return renderFunction< MeshType, RealType, 2, 0, 0 >( parameters );
   if( xDiff == 2 && yDiff == 0 && zDiff == 1 )
      return renderFunction< MeshType, RealType, 2, 0, 1 >( parameters );
   if( xDiff == 2 && yDiff == 0 && zDiff == 2 )
      return renderFunction< MeshType, RealType, 2, 0, 2 >( parameters );
   if( xDiff == 2 && yDiff == 1 && zDiff == 0 )
      return renderFunction< MeshType, RealType, 2, 1, 0 >( parameters );
   if( xDiff == 2 && yDiff == 1 && zDiff == 1 )
      return renderFunction< MeshType, RealType, 2, 1, 1 >( parameters );
   if( xDiff == 2 && yDiff == 2 && zDiff == 0 )
      return renderFunction< MeshType, RealType, 2, 2, 0 >( parameters );
   if( xDiff == 3 && yDiff == 0 && zDiff == 0 )
      return renderFunction< MeshType, RealType, 3, 0, 0 >( parameters );
   if( xDiff == 3 && yDiff == 0 && zDiff == 1 )
      return renderFunction< MeshType, RealType, 3, 0, 1 >( parameters );
   if( xDiff == 3 && yDiff == 1 && zDiff == 0 )
      return renderFunction< MeshType, RealType, 3, 1, 0 >( parameters );
   if( xDiff == 4 && yDiff == 0 && zDiff == 0 )
      return renderFunction< MeshType, RealType, 4, 0, 0 >( parameters );
   return false;
}

template< typename MeshType >
bool resolveRealType( const Config::ParameterContainer& parameters )
{
   String realType = parameters.getParameter< String >( "real-type" );
   if( realType == "mesh-real-type" )
      return resolveDerivatives< MeshType, typename MeshType::RealType >( parameters );
   if( realType == "float" )
      return resolveDerivatives< MeshType, float >( parameters );
   if( realType == "double" )
      return resolveDerivatives< MeshType, double >( parameters );
//   if( realType == "long-double" )
//      return resolveDerivatives< MeshType, long double >( parameters );
   return false;
}


template< int Dimension, typename RealType, typename IndexType >
bool resolveMesh( const std::vector< String >& parsedMeshType,
                  const Config::ParameterContainer& parameters )
{
   std::cout << "+ -> Setting mesh type to " << parsedMeshType[ 0 ] << " ... " << std::endl;
   if( parsedMeshType[ 0 ] == "Meshes::Grid" )
   {
      typedef Meshes::Grid< Dimension, RealType, Devices::Host, IndexType > MeshType;
      return resolveRealType< MeshType >( parameters );
   }
   std::cerr << "Unknown mesh type." << std::endl;
   return false;
}

template< int Dimension, typename RealType >
bool resolveIndexType( const std::vector< String >& parsedMeshType,
                       const Config::ParameterContainer& parameters )
{
   std::cout << "+ -> Setting index type to " << parsedMeshType[ 4 ] << " ... " << std::endl;

   if( parsedMeshType[ 4 ] == "int" )
      return resolveMesh< Dimension, RealType, int >( parsedMeshType, parameters );
   if( parsedMeshType[ 4 ] == "long int" )
      return resolveMesh< Dimension, RealType, long int >( parsedMeshType, parameters );

   return false;
}

template< int Dimension >
bool resolveRealType( const std::vector< String >& parsedMeshType,
                      const Config::ParameterContainer& parameters )
{
   std::cout << "+ -> Setting real type to " << parsedMeshType[ 2 ] << " ... " << std::endl;

   if( parsedMeshType[ 2 ] == "float" )
      return resolveIndexType< Dimension, float >( parsedMeshType, parameters );
   if( parsedMeshType[ 2 ] == "double" )
      return resolveIndexType< Dimension, double >( parsedMeshType, parameters );
//   if( parsedMeshType[ 2 ] == "long double" )
//      return resolveIndexType< Dimension, long double >( parsedMeshType, parameters );

   return false;
}

bool resolveMeshType( const std::vector< String >& parsedMeshType,
                      const Config::ParameterContainer& parameters )
{
   std::cout << "+ -> Setting dimensions to " << parsedMeshType[ 1 ] << " ... " << std::endl;
   int dimensions = atoi( parsedMeshType[ 1 ].getString() );

   if( dimensions == 1 )
      return resolveRealType< 1 >( parsedMeshType, parameters );
   if( dimensions == 2 )
      return resolveRealType< 2 >( parsedMeshType, parameters );
   if( dimensions == 3 )
      return resolveRealType< 3 >( parsedMeshType, parameters );

   return false;
}
