/***************************************************************************
                          tnl-init.h  -  description
                             -------------------
    begin                : Nov 23, 2013
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

#ifndef TNL_INIT_H_
#define TNL_INIT_H_

#include <config/tnlParameterContainer.h>
#include <core/vectors/tnlVector.h>
#include <mesh/tnlGrid.h>
#include <functors/tnlFunctionDiscretizer.h>
#include <functors/tnlTestFunction.h>
#include <operators/tnlFiniteDifferences.h>
#include <core/mfilename.h>

template< typename MeshType,
          typename RealType,
          int xDiff,
          int yDiff,
          int zDiff >
bool renderFunction( const tnlParameterContainer& parameters )
{
   MeshType mesh;
   tnlString meshFile = parameters.getParameter< tnlString >( "mesh" );
   cout << "+ -> Loading mesh from " << meshFile << " ... " << endl;
   if( ! mesh.load( meshFile ) )
      return false;

   typedef tnlTestFunction< MeshType::meshDimensions, RealType > FunctionType;
   FunctionType function;
   cout << "Setting up the function ... " << endl;
   if( ! function.setup( parameters, "" ) )
      return false;
   cout << "done." << endl;
   typedef tnlVector< RealType, tnlHost, typename MeshType::IndexType > DiscreteFunctionType;
   DiscreteFunctionType discreteFunction;
   if( ! discreteFunction.setSize( mesh.template getEntitiesCount< typename MeshType::Cell >() ) )
      return false;
   
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
         cout << "+ -> Computing the finite differences ... " << endl;
         DiscreteFunctionType auxDiscreteFunction;
         if( ! auxDiscreteFunction.setSize( mesh.template getEntitiesCount< typename MeshType::Cell >() ) )
            return false;
         tnlFunctionDiscretizer< MeshType, FunctionType, DiscreteFunctionType >::template discretize< 0, 0, 0 >( mesh, function, auxDiscreteFunction, time );
         tnlFiniteDifferences< MeshType >::template getDifference< DiscreteFunctionType, xDiff, yDiff, zDiff, 0, 0, 0 >( mesh, auxDiscreteFunction, discreteFunction );
      }
      else
      {
         tnlFunctionDiscretizer< MeshType, FunctionType, DiscreteFunctionType >::template discretize< xDiff, yDiff, zDiff >( mesh, function, discreteFunction, time );
      }

      tnlString outputFile = parameters.getParameter< tnlString >( "output-file" );
      if( finalTime > 0.0 )
      {
         tnlString extension = tnlString( "." ) + getFileExtension( outputFile );
         RemoveFileExtension( outputFile );
         outputFile += "-";
         tnlString aux;
         FileNameBaseNumberEnding( outputFile.getString(),
                                   step,
                                   5,
                                   extension.getString(),
                                   aux );
         outputFile = aux;
         cout << "+ -> Writing the function at the time " << time << " to " << outputFile << " ... " << endl;
      }
      else
         cout << "+ -> Writing the function to " << outputFile << " ... " << endl;
      if( ! discreteFunction.save( outputFile) )
         return false;
      time += tau;
      step ++;
   }
   return true;
}

template< typename MeshType,
          typename RealType >
bool resolveDerivatives( const tnlParameterContainer& parameters )
{

   int xDiff = parameters.getParameter< int >( "x-derivative" );
   int yDiff = parameters.getParameter< int >( "y-derivative" );
   int zDiff = parameters.getParameter< int >( "z-derivative" );
   if( xDiff < 0 || yDiff < 0 || zDiff < 0 || ( xDiff + yDiff + zDiff ) > 4 )
   {
      cerr << "Wrong orders of partial derivatives: "
           << xDiff << " " << yDiff << " " << zDiff << ". "
           << "They can be only non-negative integer numbers in sum not larger than 4."
           << endl;
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
bool resolveRealType( const tnlParameterContainer& parameters )
{
   tnlString realType = parameters.getParameter< tnlString >( "real-type" );
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
bool resolveMesh( const tnlList< tnlString >& parsedMeshType,
                  const tnlParameterContainer& parameters )
{
   cout << "+ -> Setting mesh type to " << parsedMeshType[ 0 ] << " ... " << endl;
   if( parsedMeshType[ 0 ] == "tnlGrid" )
   {
      typedef tnlGrid< Dimensions, RealType, tnlHost, IndexType > MeshType;
      return resolveRealType< MeshType >( parameters );
   }
   cerr << "Unknown mesh type." << endl;
   return false;
}

template< int Dimensions, typename RealType >
bool resolveIndexType( const tnlList< tnlString >& parsedMeshType,
                       const tnlParameterContainer& parameters )
{
   cout << "+ -> Setting index type to " << parsedMeshType[ 4 ] << " ... " << endl;
   if( parsedMeshType[ 4 ] == "int" )
      return resolveMesh< Dimensions, RealType, int >( parsedMeshType, parameters );

   if( parsedMeshType[ 4 ] == "long int" )
      return resolveMesh< Dimensions, RealType, long int >( parsedMeshType, parameters );
}

template< int Dimensions >
bool resolveRealType( const tnlList< tnlString >& parsedMeshType,
                      const tnlParameterContainer& parameters )
{
   cout << "+ -> Setting real type to " << parsedMeshType[ 2 ] << " ... " << endl;
   if( parsedMeshType[ 2 ] == "float" )
      return resolveIndexType< Dimensions, float >( parsedMeshType, parameters );

   if( parsedMeshType[ 2 ] == "double" )
      return resolveIndexType< Dimensions, double >( parsedMeshType, parameters );

   if( parsedMeshType[ 2 ] == "long-double" )
      return resolveIndexType< Dimensions, long double >( parsedMeshType, parameters );
}

bool resolveMeshType( const tnlList< tnlString >& parsedMeshType,
                      const tnlParameterContainer& parameters )
{
   cout << "+ -> Setting dimensions to " << parsedMeshType[ 1 ] << " ... " << endl;
   int dimensions = atoi( parsedMeshType[ 1 ].getString() );

   if( dimensions == 1 )
      return resolveRealType< 1 >( parsedMeshType, parameters );

   if( dimensions == 2 )
      return resolveRealType< 2 >( parsedMeshType, parameters );


   if( dimensions == 3 )
      return resolveRealType< 3 >( parsedMeshType, parameters );

}
#endif /* TNL_INIT_H_ */
