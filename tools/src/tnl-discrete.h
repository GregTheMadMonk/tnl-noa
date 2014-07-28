/***************************************************************************
                          tnl-discrete.h  -  description
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

#ifndef TNL_DISCRETE_H_
#define TNL_DISCRETE_H_

#include <config/tnlParameterContainer.h>
#include <core/vectors/tnlVector.h>
#include <mesh/tnlGrid.h>
#include <generators/functions/tnlFunctionDiscretizer.h>
#include <generators/functions/tnlSinWaveFunction.h>
#include <generators/functions/tnlSinBumpsFunction.h>
#include <generators/functions/tnlExpBumpFunction.h>
#include <schemes/tnlFiniteDifferences.h>

template< typename MeshType,
          typename FunctionType,
          int xDiff,
          int yDiff,
          int zDiff >
bool renderFunction( const tnlParameterContainer& parameters )
{
   MeshType mesh;
   tnlString meshFile = parameters.GetParameter< tnlString >( "mesh" );
   cout << "+ -> Loading mesh from " << meshFile << " ... " << endl;
   if( ! mesh.load( meshFile ) )
      return false;

   FunctionType function;
   if( ! function.init( parameters ) )
      return false;
   typedef tnlVector< typename MeshType::RealType, tnlHost, typename MeshType::IndexType > DiscreteFunctionType;
   DiscreteFunctionType discreteFunction;
   if( ! discreteFunction.setSize( mesh.getNumberOfCells() ) )
      return false;

   bool approximateDerivatives = parameters.GetParameter< bool >( "approximate-derivatives" );
   if( approximateDerivatives )
   {
      cout << "+ -> Computing the finite differences ... " << endl;
      DiscreteFunctionType auxDiscreteFunction;
      if( ! auxDiscreteFunction.setSize( mesh.getNumberOfCells() ) )
         return false;
      tnlFunctionDiscretizer< MeshType, FunctionType, DiscreteFunctionType >::template discretize< 0, 0, 0 >( mesh, function, auxDiscreteFunction );
      tnlFiniteDifferences< MeshType >::template getDifference< DiscreteFunctionType, xDiff, yDiff, zDiff, 0, 0, 0 >( mesh, auxDiscreteFunction, discreteFunction );
   }
   else
   {
      tnlFunctionDiscretizer< MeshType, FunctionType, DiscreteFunctionType >::template discretize< xDiff, yDiff, zDiff >( mesh, function, discreteFunction );
   }

   tnlString outputFile = parameters.GetParameter< tnlString >( "output-file" );
   cout << "+ -> Writing the function to " << outputFile << " ... " << endl;
   if( ! discreteFunction.save( outputFile) )
      return false;
   return true;
}

template< typename MeshType, typename FunctionType >
bool resolveDerivatives( const tnlParameterContainer& parameters )
{

   int xDiff = parameters.GetParameter< int >( "x-derivative" );
   int yDiff = parameters.GetParameter< int >( "y-derivative" );
   int zDiff = parameters.GetParameter< int >( "z-derivative" );
   if( xDiff < 0 || yDiff < 0 || zDiff < 0 || ( xDiff + yDiff + zDiff ) > 4 )
   {
      cerr << "Wrong orders of partial derivatives: "
           << xDiff << " " << yDiff << " " << zDiff << ". "
           << "They can be only non-negative integer numbers in sum not larger than 4."
           << endl;
      return false;
   }
   if( xDiff == 0 && yDiff == 0 && zDiff == 0 )
      return renderFunction< MeshType, FunctionType, 0, 0, 0 >( parameters );
   if( xDiff == 0 && yDiff == 0 && zDiff == 1 )
      return renderFunction< MeshType, FunctionType, 0, 0, 1 >( parameters );
   if( xDiff == 0 && yDiff == 0 && zDiff == 2 )
      return renderFunction< MeshType, FunctionType, 0, 0, 2 >( parameters );
   if( xDiff == 0 && yDiff == 0 && zDiff == 3 )
      return renderFunction< MeshType, FunctionType, 0, 0, 3 >( parameters );
   if( xDiff == 0 && yDiff == 0 && zDiff == 4 )
      return renderFunction< MeshType, FunctionType, 0, 0, 4 >( parameters );
   if( xDiff == 0 && yDiff == 1 && zDiff == 0 )
      return renderFunction< MeshType, FunctionType, 0, 1, 0 >( parameters );
   if( xDiff == 0 && yDiff == 1 && zDiff == 1 )
      return renderFunction< MeshType, FunctionType, 0, 1, 1 >( parameters );
   if( xDiff == 0 && yDiff == 1 && zDiff == 2 )
      return renderFunction< MeshType, FunctionType, 0, 1, 2 >( parameters );
   if( xDiff == 0 && yDiff == 1 && zDiff == 3 )
      return renderFunction< MeshType, FunctionType, 0, 1, 3 >( parameters );
   if( xDiff == 0 && yDiff == 2 && zDiff == 0 )
      return renderFunction< MeshType, FunctionType, 0, 2, 0 >( parameters );
   if( xDiff == 0 && yDiff == 2 && zDiff == 1 )
      return renderFunction< MeshType, FunctionType, 0, 2, 1 >( parameters );
   if( xDiff == 0 && yDiff == 2 && zDiff == 2 )
      return renderFunction< MeshType, FunctionType, 0, 2, 2 >( parameters );
   if( xDiff == 0 && yDiff == 3 && zDiff == 0 )
      return renderFunction< MeshType, FunctionType, 0, 3, 0 >( parameters );
   if( xDiff == 0 && yDiff == 3 && zDiff == 1 )
      return renderFunction< MeshType, FunctionType, 0, 3, 1 >( parameters );
   if( xDiff == 0 && yDiff == 4 && zDiff == 0 )
      return renderFunction< MeshType, FunctionType, 0, 4, 0 >( parameters );
   if( xDiff == 1 && yDiff == 0 && zDiff == 0 )
      return renderFunction< MeshType, FunctionType, 1, 0, 0 >( parameters );
   if( xDiff == 1 && yDiff == 0 && zDiff == 1 )
      return renderFunction< MeshType, FunctionType, 1, 0, 1 >( parameters );
   if( xDiff == 1 && yDiff == 0 && zDiff == 2 )
      return renderFunction< MeshType, FunctionType, 1, 0, 2 >( parameters );
   if( xDiff == 1 && yDiff == 0 && zDiff == 3 )
      return renderFunction< MeshType, FunctionType, 1, 0, 3 >( parameters );
   if( xDiff == 1 && yDiff == 1 && zDiff == 0 )
      return renderFunction< MeshType, FunctionType, 1, 1, 0 >( parameters );
   if( xDiff == 1 && yDiff == 1 && zDiff == 1 )
      return renderFunction< MeshType, FunctionType, 1, 1, 1 >( parameters );
   if( xDiff == 1 && yDiff == 1 && zDiff == 2 )
      return renderFunction< MeshType, FunctionType, 1, 1, 2 >( parameters );
   if( xDiff == 1 && yDiff == 2 && zDiff == 0 )
      return renderFunction< MeshType, FunctionType, 1, 2, 0 >( parameters );
   if( xDiff == 1 && yDiff == 2 && zDiff == 1 )
      return renderFunction< MeshType, FunctionType, 1, 2, 1 >( parameters );
   if( xDiff == 1 && yDiff == 3 && zDiff == 0 )
      return renderFunction< MeshType, FunctionType, 1, 3, 0 >( parameters );
   if( xDiff == 2 && yDiff == 0 && zDiff == 0 )
      return renderFunction< MeshType, FunctionType, 2, 0, 0 >( parameters );
   if( xDiff == 2 && yDiff == 0 && zDiff == 1 )
      return renderFunction< MeshType, FunctionType, 2, 0, 1 >( parameters );
   if( xDiff == 2 && yDiff == 0 && zDiff == 2 )
      return renderFunction< MeshType, FunctionType, 2, 0, 2 >( parameters );
   if( xDiff == 2 && yDiff == 1 && zDiff == 0 )
      return renderFunction< MeshType, FunctionType, 2, 1, 0 >( parameters );
   if( xDiff == 2 && yDiff == 1 && zDiff == 1 )
      return renderFunction< MeshType, FunctionType, 2, 1, 1 >( parameters );
   if( xDiff == 2 && yDiff == 2 && zDiff == 0 )
      return renderFunction< MeshType, FunctionType, 2, 2, 0 >( parameters );
   if( xDiff == 3 && yDiff == 0 && zDiff == 0 )
      return renderFunction< MeshType, FunctionType, 3, 0, 0 >( parameters );
   if( xDiff == 3 && yDiff == 0 && zDiff == 1 )
      return renderFunction< MeshType, FunctionType, 3, 0, 1 >( parameters );
   if( xDiff == 3 && yDiff == 1 && zDiff == 0 )
      return renderFunction< MeshType, FunctionType, 3, 1, 0 >( parameters );
   if( xDiff == 4 && yDiff == 0 && zDiff == 0 )
      return renderFunction< MeshType, FunctionType, 4, 0, 0 >( parameters );
   return false;
}


template< typename MeshType >
bool resolveFunction( const tnlParameterContainer& parameters )
{
   tnlString functionName = parameters.GetParameter< tnlString >( "function" );
   cout << "+ -> Generating function " << functionName << " ... " << endl;
   if( functionName == "sin-wave" )
   {
      typedef tnlSinWaveFunction< MeshType::Dimensions, typename MeshType::VertexType, typename MeshType::DeviceType > FunctionType;
      return resolveDerivatives< MeshType, FunctionType >( parameters );
   }
   if( functionName == "sin-bumps" )
   {
      typedef tnlSinBumpsFunction< MeshType::Dimensions, typename MeshType::VertexType, typename MeshType::DeviceType > FunctionType;
      return resolveDerivatives< MeshType, FunctionType >( parameters );
   }
   if( functionName == "exp-bump" )
   {
      typedef tnlExpBumpFunction< MeshType::Dimensions, typename MeshType::VertexType, typename MeshType::DeviceType > FunctionType;
      return resolveDerivatives< MeshType, FunctionType >( parameters );
   }
   cerr << "Unknown function " << functionName << "." << endl;
   return false;
}

template< int Dimensions, typename RealType, typename IndexType >
bool resolveMesh( const tnlList< tnlString >& parsedMeshType,
                  const tnlParameterContainer& parameters )
{
   cout << "+ -> Setting mesh type to " << parsedMeshType[ 0 ] << " ... " << endl;
   if( parsedMeshType[ 0 ] == "tnlGrid" )
   {
      tnlList< tnlString > parsedGeometryType;
      if( ! parseObjectType( parsedMeshType[ 5 ], parsedGeometryType ) )
      {
         cerr << "Unable to parse the geometry type " << parsedMeshType[ 5 ] << "." << endl;
         return false;
      }
      typedef tnlGrid< Dimensions, RealType, tnlHost, IndexType > MeshType;
      return resolveFunction< MeshType >( parameters );
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
#endif /* TNL_DISCRETE_H_ */
