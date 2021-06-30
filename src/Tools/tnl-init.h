/***************************************************************************
                          tnl-init.h  -  description
                             -------------------
    begin                : Nov 23, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/MPI/Wrappers.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Meshes/Readers/VTIReader.h>
#include <TNL/Meshes/Readers/PVTIReader.h>
#include <TNL/Meshes/Writers/VTIWriter.h>
#include <TNL/Meshes/Writers/PVTIWriter.h>
#include <TNL/Functions/TestFunction.h>
#include <TNL/Operators/FiniteDifferences.h>
#include <TNL/FileName.h>
#include <TNL/Functions/MeshFunction.h>

using namespace TNL;

template< typename MeshType,
          typename RealType,
          int xDiff,
          int yDiff,
          int zDiff >
bool renderFunction( const Config::ParameterContainer& parameters )
{
   using namespace  Meshes::DistributedMeshes;
   using DistributedGridType = Meshes::DistributedMeshes::DistributedMesh<MeshType>;
   DistributedGridType distributedMesh;
   Pointers::SharedPointer< MeshType > meshPointer;

   const String meshFile = parameters.getParameter< String >( "mesh" );
   std::cout << "+ -> Loading mesh from " << meshFile << " ... " << std::endl;

   if( TNL::MPI::GetSize() > 1 )
   {
      Meshes::Readers::PVTIReader reader( meshFile );
      reader.loadMesh( distributedMesh );
      *meshPointer = distributedMesh.getLocalMesh();
   }
   else
   {
      Meshes::Readers::VTIReader reader( meshFile );
      reader.loadMesh( *meshPointer );
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

      const std::string meshFunctionName = parameters.getParameter< std::string >( "mesh-function-name" );

      if( TNL::MPI::GetSize() > 1 )
      {
         std::ofstream file;
         if( TNL::MPI::GetRank() == 0 )
            file.open( outputFile );
         using PVTI = Meshes::Writers::PVTIWriter< typename DistributedGridType::GridType >;
         PVTI pvti( file );
         // TODO: write metadata: step and time
         pvti.writeImageData( distributedMesh );
         // TODO
         //if( distributedMesh.getGhostLevels() > 0 ) {
         //   pvti.template writePPointData< std::uint8_t >( Meshes::VTK::ghostArrayName() );
         //   pvti.template writePCellData< std::uint8_t >( Meshes::VTK::ghostArrayName() );
         //}
         if( meshFunction->getEntitiesDimension() == 0 )
            pvti.template writePPointData< typename MeshFunctionType::RealType >( meshFunctionName );
         else
            pvti.template writePCellData< typename MeshFunctionType::RealType >( meshFunctionName );
         const std::string subfilePath = pvti.addPiece( outputFile, distributedMesh );

         // create a .vti file for local data
         // TODO: write metadata: step and time
         using Writer = Meshes::Writers::VTIWriter< typename DistributedGridType::GridType >;
         std::ofstream subfile( subfilePath );
         Writer writer( subfile );
         // NOTE: passing the local mesh to writeImageData does not work correctly, just like meshFunction->write(...)
         //       (it does not write the correct extent of the subdomain - globalBegin is only in the distributed grid)
         // NOTE: globalBegin and globalEnd here are without overlaps
         writer.writeImageData( distributedMesh.getGlobalGrid().getOrigin(),
                                distributedMesh.getGlobalBegin(),
                                distributedMesh.getGlobalBegin() + distributedMesh.getLocalSize(),
                                distributedMesh.getGlobalGrid().getSpaceSteps() );
         if( meshFunction->getEntitiesDimension() == 0 )
            writer.writePointData( meshFunction->getData(), meshFunctionName );
         else
            writer.writeCellData( meshFunction->getData(), meshFunctionName );
         // TODO
         //if( mesh.getGhostLevels() > 0 ) {
         //   writer.writePointData( mesh.vtkPointGhostTypes(), Meshes::VTK::ghostArrayName() );
         //   writer.writeCellData( mesh.vtkCellGhostTypes(), Meshes::VTK::ghostArrayName() );
         //}
      }
      else
      {
         // TODO: write metadata: step and time
         meshFunction->write( meshFunctionName, outputFile, "auto" );
      }

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
