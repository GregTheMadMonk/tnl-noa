/***************************************************************************
                          tnl-lattice-init.cpp  -  description
                             -------------------
    begin                : Jun 13, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ParameterContainer.h>

template< typename Mesh, typename Real >
bool resolveProfileMesh( const Config::ParameterContainer& parameters )
{
   
}

template< typename MeshType >
bool resolveRealType( const Config::ParameterContainer& parameters )
{
   String realType = parameters.getParameter< String >( "real-type" );
   if( realType == "mesh-real-type" )
      return resolveProfileMesh< MeshType, typename MeshType::RealType >( parameters );
   if( realType == "float" )
      return resolveProfileMesh< MeshType, float >( parameters );
   if( realType == "double" )
      return resolveProfileMesh< MeshType, double >( parameters );
   if( realType == "long-double" )
      return resolveProfileMesh< MeshType, long double >( parameters );
   return false;
}

template< int Dimension, typename RealType, typename IndexType >
bool resolveMesh( const Containers::List< String >& parsedMeshType,
                  const Config::ParameterContainer& parameters )
{
  std::cout << "+ -> Setting mesh type to " << parsedMeshType[ 0 ] << " ... " << std::endl;
   if( parsedMeshType[ 0 ] == "Meshes::Grid" ||
       parsedMeshType[ 0 ] == "tnlGrid" )  // TODO: remove deprecated type name
   {
      typedef Meshes::Grid< Dimension, RealType, Devices::Host, IndexType > MeshType;
      return resolveRealType< MeshType >( parameters );
   }
   std::cerr << "Unknown mesh type." << std::endl;
   return false;
}

template< int Dimension, typename RealType >
bool resolveIndexType( const Containers::List< String >& parsedMeshType,
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
bool resolveRealType( const Containers::List< String >& parsedMeshType,
                      const Config::ParameterContainer& parameters )
{
  std::cout << "+ -> Setting real type to " << parsedMeshType[ 2 ] << " ... " << std::endl;
   if( parsedMeshType[ 2 ] == "float" )
      return resolveIndexType< Dimension, float >( parsedMeshType, parameters );

   if( parsedMeshType[ 2 ] == "double" )
      return resolveIndexType< Dimension, double >( parsedMeshType, parameters );

   if( parsedMeshType[ 2 ] == "long-double" )
      return resolveIndexType< Dimension, long double >( parsedMeshType, parameters );

   return false;
}

bool resolveMeshType( const Containers::List< String >& parsedMeshType,
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