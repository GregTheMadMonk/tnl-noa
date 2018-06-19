/***************************************************************************
                          tnl-lattice-init.cpp  -  description
                             -------------------
    begin                : Jun 13, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Object.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridEntity.h>
#include <TNL/Functions/MeshFunction.h>

using namespace TNL;

template< typename MeshFunction, typename ProfileMeshFunction >
bool performExtrude( const Config::ParameterContainer& parameters,
                     MeshFunction& f,
                     const ProfileMeshFunction& profile )
{
   using MeshPointer = SharedPointer< typename MeshFunction::MeshType >;
   using ProfileMeshPointer = SharedPointer< typename ProfileMeshFunction::MeshType >;
   using ProfileMeshType = typename ProfileMeshFunction::MeshType;
   using MeshType = typename MeshFunction::MeshType;
   using RealType = typename MeshFunction::RealType;
   using IndexType = typename MeshType::IndexType;
   using CellType = typename MeshType::Cell;
   using PointType = typename MeshType::PointType;
   using ProfilePointType = typename ProfileMeshType::PointType;
   using ProfileCellType = typename ProfileMeshType::Cell;
   String profileOrientation = parameters.getParameter< String >( "profile-orientation" );
   if( profileOrientation != "x" && 
       profileOrientation != "y" &&
       profileOrientation != "z" )
   {
      std::cerr << "Wrong profile orientation " << profileOrientation << "." << std::endl;
      return false;
   }
   double profileShiftX = parameters.getParameter< double >( "profile-shift-x" );
   double profileShiftY = parameters.getParameter< double >( "profile-shift-y" );
   double profileShiftZ = parameters.getParameter< double >( "profile-shift-z" );
   double profileScaleX = parameters.getParameter< double >( "profile-scale-x" );
   double profileScaleY = parameters.getParameter< double >( "profile-scale-y" );
   double profileScaleZ = parameters.getParameter< double >( "profile-scale-z" );   
   double profileRotation = parameters.getParameter< double >( "profile-rotation" );   
   double extrudeStart = parameters.getParameter< double >( "extrude-start" );
   double extrudeStop = parameters.getParameter< double >( "extrude-stop" );
   const MeshType mesh = f.getMesh();
   ProfileMeshType profileMesh = profile.getMesh();
   CellType cell( mesh );
   IndexType& i = cell.getCoordinates().x();
   IndexType& j = cell.getCoordinates().y();
   IndexType& k = cell.getCoordinates().z();
   ProfilePointType profileCenter( profileMesh.getOrigin() + 0.5 * profileMesh.getProportions() );
   const RealType rotationSin = sin( M_PI * profileRotation / 180.0 );
   const RealType rotationCos = cos( M_PI * profileRotation / 180.0 );
   for( i = 0; i < mesh.getDimensions().x(); i++ )
      for( j = 0; j < mesh.getDimensions().y(); j++ )
         for( k = 0; k < mesh.getDimensions().z(); k++ )
         {
            cell.refresh();
            PointType p = cell.getCenter();
            p.x() /= profileScaleX;
            p.y() /= profileScaleY;
            p.z() /= profileScaleZ;
            p.x() -= profileShiftX;
            p.y() -= profileShiftY;
            p.z() -= profileShiftZ;
            if( profileOrientation == "x" )
            {
               if( p.z() < profileMesh.getOrigin().x() ||
                   p.z() > profileMesh.getOrigin().x() + profileMesh.getProportions().x() ||
                   p.y() < profileMesh.getOrigin().y() ||
                   p.y() > profileMesh.getOrigin().y() + profileMesh.getProportions().y() )
                  continue;
               if( p.x() < extrudeStart || p.x() > extrudeStop )
                  f( cell ) = 0.0;
               else
               {
                  ProfileCellType profileCell( profileMesh );
                  ProfilePointType aux1( ( p.z() - profileMesh.getOrigin().x() ),
                                 ( p.y() - profileMesh.getOrigin().y() ) );
                  aux1 -= profileCenter;
                  ProfilePointType aux2( rotationCos * aux1.x() - rotationSin * aux1.y(),
                                  rotationSin * aux1.x() + rotationCos * aux1.y() );
                  aux1 = profileCenter + aux2;
                  profileCell.getCoordinates().x() = aux1.x() / profileMesh.getSpaceSteps().x();
                  profileCell.getCoordinates().y() = aux1.y() / profileMesh.getSpaceSteps().y();
                  profileCell.refresh();
                  RealType aux = profile( profileCell );
                  if( aux ) f( cell ) = aux;
               }
            }
            if( profileOrientation == "y" )
            {
               if( p.x() < profileMesh.getOrigin().x() ||
                   p.x() > profileMesh.getOrigin().x() + profileMesh.getProportions().x() ||
                   p.z() < profileMesh.getOrigin().y() ||
                   p.z() > profileMesh.getOrigin().y() + profileMesh.getProportions().y() )
                  continue;
               if( p.y() < extrudeStart || p.y() > extrudeStop )
                  f( cell ) = 0.0;
               else
               {
                  ProfileCellType profileCell( profileMesh );
                  ProfilePointType aux1( ( p.x() - profileMesh.getOrigin().x() ),
                                 ( p.z() - profileMesh.getOrigin().y() ) );
                  aux1 -= profileCenter;
                  ProfilePointType aux2( rotationCos * aux1.x() - rotationSin * aux1.y(),
                                  rotationSin * aux1.x() + rotationCos * aux1.y() );
                  aux1 = profileCenter + aux2;
                  profileCell.getCoordinates().x() = aux1.x() / profileMesh.getSpaceSteps().x();
                  profileCell.getCoordinates().y() = aux1.y() / profileMesh.getSpaceSteps().y();
                  profileCell.refresh();
                  RealType aux = profile( profileCell );
                  if( aux ) f( cell ) = aux;
               }
            }            
            if( profileOrientation == "z" )
            {
               if( p.x() < profileMesh.getOrigin().x() ||
                   p.x() > profileMesh.getOrigin().x() + profileMesh.getProportions().x() ||
                   p.y() < profileMesh.getOrigin().y() ||
                   p.y() > profileMesh.getOrigin().y() + profileMesh.getProportions().y() )
                  continue;
               if( p.z() < extrudeStart || p.z() > extrudeStop )
                  f( cell ) = 0.0;
               else
               {
                  ProfileCellType profileCell( profileMesh );
                  ProfilePointType aux1( ( p.x() - profileMesh.getOrigin().x() ),
                                 ( p.y() - profileMesh.getOrigin().y() ) );
                  aux1 -= profileCenter;
                  ProfilePointType aux2( rotationCos * aux1.x() - rotationSin * aux1.y(),
                                  rotationSin * aux1.x() + rotationCos * aux1.y() );
                  aux1 = profileCenter + aux2;
                  profileCell.getCoordinates().x() = aux1.x() / profileMesh.getSpaceSteps().x();
                  profileCell.getCoordinates().y() = aux1.y() / profileMesh.getSpaceSteps().y();
                  profileCell.refresh();
                  RealType aux = profile( profileCell );
                  if( aux ) f( cell ) = aux;
               }
            }
         }
   String outputFile = parameters.getParameter< String >( "output-file" );
   if( ! f.save( outputFile ) )
   {
      std::cerr << "Unable to save output file " << outputFile << "." << std::endl;
      return false;
   }
   return true;
}


template< typename Real, typename Mesh, typename ProfileMeshFunction >
bool
readProfileMeshFunction( const Config::ParameterContainer& parameters )
{
   String profileMeshFile = parameters.getParameter< String >( "profile-mesh" );
   using ProfileMeshPointer = SharedPointer< typename ProfileMeshFunction::MeshType >;
   ProfileMeshPointer profileMesh;
   if( ! profileMesh->load( profileMeshFile ) )
   {
      std::cerr << "Unable to load the profile mesh file." << profileMeshFile << "." << std::endl;
      return false;
   }
   String profileFile = parameters.getParameter< String >( "profile-file" );
   ProfileMeshFunction profileMeshFunction( profileMesh );
   if( ! profileMeshFunction.load( profileFile ) )
   {
      std::cerr << "Unable to load profile mesh function from the file " << profileFile << "." << std::endl;
      return false;
   }
   String meshFile = parameters.getParameter< String >( "mesh" );
   using MeshPointer = SharedPointer< Mesh >;
   MeshPointer mesh;
   if( ! mesh->load( meshFile ) )
   {
      std::cerr << "Unable to load 3D mesh from the file " << meshFile << "." << std::endl;
      return false;
   }
   using MeshFunction = Functions::MeshFunction< Mesh, 3, Real >;
   MeshFunction meshFunction( mesh );
   if( parameters.checkParameter( "input-file" ) )
   {
      const String& inputFile = parameters.getParameter< String >( "input-file" ); 
      if( ! meshFunction.load( inputFile ) )
      {
         std::cerr << "Unable to load " << inputFile << "." << std::endl;
         return false;
      }
   }
   else meshFunction.getData().setValue( 0.0 );
   if( parameters.getParameter< String >( "operation" ) == "extrude" )
      performExtrude( parameters, meshFunction, profileMeshFunction );
   return true;
}

template< typename ProfileMesh, typename Real, typename Mesh >
bool resolveProfileReal( const Config::ParameterContainer& parameters )
{
   String profileFile = parameters. getParameter< String >( "profile-file" );
   String meshFunctionType;
   if( ! getObjectType( profileFile, meshFunctionType ) )
   {
      std::cerr << "I am not able to detect the mesh function type from the profile file " << profileFile << "." << std::endl;
      return EXIT_FAILURE;
   }
   //std::cout << meshFunctionType << " detected in " << profileFile << " file." << std::endl;
   Containers::List< String > parsedMeshFunctionType;
   if( ! parseObjectType( meshFunctionType, parsedMeshFunctionType ) )
   {
      std::cerr << "Unable to parse the mesh function type " << meshFunctionType << "." << std::endl;
      return EXIT_FAILURE;
   }
   //std::cout << parsedMeshFunctionType << std::endl;
   if( parsedMeshFunctionType[ 0 ] != "Functions::MeshFunction" )
   {
      std::cerr << "MeshFunction is required in profile file " << profileFile << "." << std::endl;
      return false;
   }
   if( parsedMeshFunctionType[ 1 ] != ProfileMesh::getType() )
   {
      std::cerr << "The mesh function in the profile file must be defined on " << ProfileMesh::getType() 
                << " but it is defined on " << parsedMeshFunctionType[ 1 ] << "." << std::endl;
      return false;
   }
   if( parsedMeshFunctionType[ 2 ] != "2" )
   {
      std::cerr << "The mesh function must be defined on cells but it is defined on mesh entities with " << parsedMeshFunctionType[ 2 ] << " dimensions." << std::endl;
      return false;
   }
   if( parsedMeshFunctionType[ 3 ] == "float" )
      return readProfileMeshFunction< Real, Mesh, Functions::MeshFunction< ProfileMesh, 2, float > >( parameters );
   if( parsedMeshFunctionType[ 3 ] == "double" )
      return readProfileMeshFunction< Real, Mesh, Functions::MeshFunction< ProfileMesh, 2, double > >( parameters );
   std::cerr << "Unknown real type " << parsedMeshFunctionType[ 3 ] << " of mesh function in the file " << profileFile << "." << std::endl;
   return false;
}

template< typename ProfileMesh, typename Real, typename MeshReal >
bool resolveMeshIndexType( const Containers::List< String >& parsedMeshType,
                           const Config::ParameterContainer& parameters )
{
   if( parsedMeshType[ 4 ] == "int" )
      return resolveProfileReal< ProfileMesh, Real, Meshes::Grid< 3, MeshReal, Devices::Host, int > >( parameters );

   if( parsedMeshType[ 4 ] == "long int" )
      return resolveProfileReal< ProfileMesh, Real, Meshes::Grid< 3, MeshReal, Devices::Host, long int > >( parameters );  

   std::cerr << "Unknown index type " << parsedMeshType[ 4 ] << "." << std::endl;
   return false;
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
   if( dimensions != 3 )
   {
      std::cerr << "The main mesh '" << meshFile << "' must be a 3D grid." << std::endl;
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
      return resolveProfileMeshIndexType< float >( parsedMeshType, parameters );

   if( parsedMeshType[ 2 ] == "double" )
      return resolveProfileMeshIndexType< double >( parsedMeshType, parameters );

   if( parsedMeshType[ 2 ] == "long-double" )
      return resolveProfileMeshIndexType< long double >( parsedMeshType, parameters );

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
