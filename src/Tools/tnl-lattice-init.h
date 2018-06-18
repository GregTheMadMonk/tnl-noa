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

using namespace TNL;

template< typename ProfileMesh, typename Resl, typename Mesh >
bool resolveProfileReal( const Config::ParameterContainer& parameters )
{
   String profileFile = parameters. getParameter< String >( "profile-file" );
   String meshFunctionType;
   if( ! getObjectType( profileFile, meshFunctionType ) )
   {
      std::cerr << "I am not able to detect the mesh function type from the profile file " << profileFile << "." << std::endl;
      return EXIT_FAILURE;
   }
   std::cout << meshFunctionType << " detected in " << profileFile << " file." << std::endl;
   Containers::List< String > parsedMeshFunctionType;
   if( ! parseObjectType( meshFunctionType, parsedMeshFunctionType ) )
   {
      std::cerr << "Unable to parse the mesh function type " << meshFunctionType << "." << std::endl;
      return EXIT_FAILURE;
   }
   
   
}

template< typename ProfileMesh, typename Real, typename MeshReal >
bool resolveMeshIndexType( const Containers::List< String >& parsedMeshType,
                                  const Config::ParameterContainer& parameters )
{
   if( parsedMeshType[ 4 ] == "int" )
      return resolveProfileReal< ProfileMesh, Real, Meshes::Grid< 3, MeshReal, Devices::Host, int >( parameters );

   if( parsedMeshType[ 4 ] == "long int" )
      return resolveProfileReal< ProfileMesh, Real, Meshes::Grid< 3, MeshReal, Device::Host, long int >( parameters );  
}

template< typename ProfileMesh, typename Real >
bool resolveMesh( const Config::ParameterContainer& parameters )
{
   String meshFile = parameters.getParameter< String >( "mesh" );
   String meshType;
   if( ! getObjectType( meshFile, meshType ) )
   {
      std::cerr << "I am not able to detect the mesh type from the file " << meshFile << "." << std::endl;
      return EXIT_FAILURE;
   }
   std::cout << meshType << " detected in " << meshFile << " file." << std::endl;
   Containers::List< String > parsedMeshType;
   if( ! parseObjectType( meshType, parsedMeshType ) )
   {
      std::cerr << "Unable to parse the mesh type " << meshType << "." << std::endl;
      return EXIT_FAILURE;
   }
   
   int dimensions = atoi( parsedMeshType[ 1 ].getString() );
   if( dimensions != 2 )
   {
      std::cerr << "The profile mesh '" << meshFile << "' must be a 2D grid." << std::endl;
      return false;
   }
   
   std::cout << "+ -> Setting real type to " << parsedMeshType[ 2 ] << " ... " << std::endl;
   if( parsedMeshType[ 2 ] == "float" )
      return resolveMeshIndexType< ProfileMesh, Real, float >( parsedMeshType, parameters );

   if( parsedMeshType[ 2 ] == "double" )
      return resolveMeshIndexType< ProfileMesh, Real, double >( parsedMeshType, parameters );

   if( parsedMeshType[ 2 ] == "long-double" )
      return resolveMeshIndexType< ProfileMesh, Real, long double >( parsedMeshType, parameters );
   return false;   
}

template< typename ProfileMesh >
bool resolveRealType( const Config::ParameterContainer& parameters )
{
   String realType = parameters.getParameter< String >( "real-type" );
   if( realType == "mesh-real-type" )
      return resolveMesh< ProfileMesh, typename ProfileMesh::RealType >( parameters );
   if( realType == "float" )
      return resolveMesh< ProfileMesh, float >( parameters );
   if( realType == "double" )
      return resolveMesh< ProfileMesh, double >( parameters );
   if( realType == "long-double" )
      return resolveMesh< ProfileMesh, long double >( parameters );
   return false;
}

template< typename RealType, typename IndexType >
bool resolveProfileMesh( const Containers::List< String >& parsedMeshType,
                  const Config::ParameterContainer& parameters )
{
  std::cout << "+ -> Setting mesh type to " << parsedMeshType[ 0 ] << " ... " << std::endl;
   if( parsedMeshType[ 0 ] == "Meshes::Grid" )
   {
      typedef Meshes::Grid< 2, RealType, Devices::Host, IndexType > MeshType;
      return resolveRealType< MeshType >( parameters );
   }
   std::cerr << "Unknown mesh type." << std::endl;
   return false;
}

template< typename RealType >
bool resolveProfileMeshIndexType( const Containers::List< String >& parsedMeshType,
                                  const Config::ParameterContainer& parameters )
{
  std::cout << "+ -> Setting index type to " << parsedMeshType[ 4 ] << " ... " << std::endl;
   if( parsedMeshType[ 4 ] == "int" )
      return resolveProfileMesh< RealType, int >( parsedMeshType, parameters );

   if( parsedMeshType[ 4 ] == "long int" )
      return resolveProfileMesh< RealType, long int >( parsedMeshType, parameters );

   return false;
}

bool resolveProfileMeshRealType( const Containers::List< String >& parsedMeshType,
                                 const Config::ParameterContainer& parameters )
{
   std::cout << "+ -> Setting real type to " << parsedMeshType[ 2 ] << " ... " << std::endl;
   if( parsedMeshType[ 2 ] == "float" )
      return resolveIndexType< float >( parsedMeshType, parameters );

   if( parsedMeshType[ 2 ] == "double" )
      return resolveIndexType< double >( parsedMeshType, parameters );

   if( parsedMeshType[ 2 ] == "long-double" )
      return resolveIndexType< long double >( parsedMeshType, parameters );

   return false;
}

bool resolveProfileMeshType( const Config::ParameterContainer& parameters )
{
   String meshFile = parameters. getParameter< String >( "profile-mesh" );
   String meshType;
   if( ! getObjectType( meshFile, meshType ) )
   {
      std::cerr << "I am not able to detect the mesh type from the file " << meshFile << "." << std::endl;
      return EXIT_FAILURE;
   }
   std::cout << meshType << " detected in " << meshFile << " file." << std::endl;
   Containers::List< String > parsedMeshType;
   if( ! parseObjectType( meshType, parsedMeshType ) )
   {
      std::cerr << "Unable to parse the mesh type " << meshType << "." << std::endl;
      return EXIT_FAILURE;
   }
   
   int dimensions = atoi( parsedMeshType[ 1 ].getString() );
   if( dimensions != 2 )
   {
      std::cerr << "The profile mesh '" << meshFile << "' must be 2D grid." << std::endl;
      return false;
   }
   return resolveProfileMeshRealType( parsedMeshType, parameters );
}