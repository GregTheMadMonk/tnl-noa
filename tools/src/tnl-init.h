/***************************************************************************
                          tnl-init.h  -  description
                             -------------------
    begin                : Nov 23, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNL_INIT_H_
#define TNL_INIT_H_

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Vectors/Vector.h>
#include <TNL/mesh/tnlGrid.h>
#include <TNL/Functions/tnlTestFunction.h>
#include <TNL/operators/tnlFiniteDifferences.h>
#include <TNL/core/mfilename.h>
#include <TNL/Functions/tnlMeshFunction.h>

using namespace TNL;

template< typename MeshType,
          typename RealType,
          int xDiff,
          int yDiff,
          int zDiff >
bool renderFunction( const Config::ParameterContainer& parameters )
{
   MeshType mesh;
   String meshFile = parameters.getParameter< String >( "mesh" );
  std::cout << "+ -> Loading mesh from " << meshFile << " ... " << std::endl;
   if( ! mesh.load( meshFile ) )
      return false;

   typedef tnlTestFunction< MeshType::meshDimensions, RealType > FunctionType;
   FunctionType function;
  std::cout << "Setting up the function ... " << std::endl;
   if( ! function.setup( parameters, "" ) )
      return false;
  std::cout << "done." << std::endl;
   typedef tnlMeshFunction< MeshType, MeshType::meshDimensions > MeshFunctionType;
   MeshFunctionType meshFunction( mesh );
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
         //tnlFiniteDifferences< MeshType >::template getDifference< DiscreteFunctionType, xDiff, yDiff, zDiff, 0, 0, 0 >( mesh, auxDiscreteFunction, discreteFunction );
      }
      else
      {
         tnlMeshFunctionEvaluator< MeshFunctionType, FunctionType >::evaluate( meshFunction, function, time );
      }

      String outputFile = parameters.getParameter< String >( "output-file" );
      if( finalTime > 0.0 )
      {
         String extension = String( "." ) + getFileExtension( outputFile );
         RemoveFileExtension( outputFile );
         outputFile += "-";
         String aux;
         FileNameBaseNumberEnding( outputFile.getString(),
                                   step,
                                   5,
                                   extension.getString(),
                                   aux );
         outputFile = aux;
        std::cout << "+ -> Writing the function at the time " << time << " to " << outputFile << " ... " << std::endl;
      }
      else
        std::cout << "+ -> Writing the function to " << outputFile << " ... " << std::endl;
      if( ! meshFunction.save( outputFile) )
         return false;
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
   if( realType == "long-double" )
      return resolveDerivatives< MeshType, long double >( parameters );
}


template< int Dimensions, typename RealType, typename IndexType >
bool resolveMesh( const List< String >& parsedMeshType,
                  const Config::ParameterContainer& parameters )
{
  std::cout << "+ -> Setting mesh type to " << parsedMeshType[ 0 ] << " ... " << std::endl;
   if( parsedMeshType[ 0 ] == "tnlGrid" )
   {
      typedef tnlGrid< Dimensions, RealType, Devices::Host, IndexType > MeshType;
      return resolveRealType< MeshType >( parameters );
   }
   std::cerr << "Unknown mesh type." << std::endl;
   return false;
}

template< int Dimensions, typename RealType >
bool resolveIndexType( const List< String >& parsedMeshType,
                       const Config::ParameterContainer& parameters )
{
  std::cout << "+ -> Setting index type to " << parsedMeshType[ 4 ] << " ... " << std::endl;
   if( parsedMeshType[ 4 ] == "int" )
      return resolveMesh< Dimensions, RealType, int >( parsedMeshType, parameters );

   if( parsedMeshType[ 4 ] == "long int" )
      return resolveMesh< Dimensions, RealType, long int >( parsedMeshType, parameters );
}

template< int Dimensions >
bool resolveRealType( const List< String >& parsedMeshType,
                      const Config::ParameterContainer& parameters )
{
  std::cout << "+ -> Setting real type to " << parsedMeshType[ 2 ] << " ... " << std::endl;
   if( parsedMeshType[ 2 ] == "float" )
      return resolveIndexType< Dimensions, float >( parsedMeshType, parameters );

   if( parsedMeshType[ 2 ] == "double" )
      return resolveIndexType< Dimensions, double >( parsedMeshType, parameters );

   if( parsedMeshType[ 2 ] == "long-double" )
      return resolveIndexType< Dimensions, long double >( parsedMeshType, parameters );
}

bool resolveMeshType( const List< String >& parsedMeshType,
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

}
#endif /* TNL_INIT_H_ */
