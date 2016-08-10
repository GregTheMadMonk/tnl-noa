/***************************************************************************
                          MeshTypeResolver_impl.h  -  description
                             -------------------
    begin                : Nov 28, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/String.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/DummyMesh.h>
#include <TNL/Solvers/SolverStarter.h>

namespace TNL {
namespace Solvers {   

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag,
          bool MeshTypeSupported = ConfigTagMesh< ConfigTag, MeshType >::enabled >
class MeshResolverTerminator{};


template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
bool MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, false  >::run( const Config::ParameterContainer& parameters )
{
   return ProblemSetter< Real,
                         Device,
                         Index,
                         Meshes::DummyMesh< Real, Device, Index >,
                         ConfigTag,
                         SolverStarter< ConfigTag > >::template run< Real, Device, Index, ConfigTag >( parameters );
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
bool MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::run( const Config::ParameterContainer& parameters )
{
   const String& meshFileName = parameters.getParameter< String >( "mesh" );

   String meshType;
   if( ! getObjectType( meshFileName, meshType ) )
   {
      std::cerr << "I am not able to detect the mesh type from the file " << meshFileName << "." << std::endl;
      return EXIT_FAILURE;
   }
  std::cout << meshType << " detected in " << meshFileName << " file." << std::endl;
   List< String > parsedMeshType;
   if( ! parseObjectType( meshType, parsedMeshType ) )
   {
      std::cerr << "Unable to parse the mesh type " << meshType << "." << std::endl;
      return false;
   }
   return resolveMeshDimensions( parameters, parsedMeshType );
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
bool MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::resolveMeshDimensions( const Config::ParameterContainer& parameters,
                                                                                                        const List< String >& parsedMeshType )
{
   int dimensions = atoi( parsedMeshType[ 1 ].getString() );

   if( dimensions == 1 )
      return resolveMeshRealType< 1 >( parameters, parsedMeshType );
   if( dimensions == 2 )
      return resolveMeshRealType< 2 >( parameters, parsedMeshType );
   if( dimensions == 3 )
      return resolveMeshRealType< 3 >( parameters, parsedMeshType );
   std::cerr << "Dimensions higher than 3 are not supported." << std::endl;
   return false;
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
   template< int MeshDimensions, typename, typename >
bool MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::resolveMeshRealType( const Config::ParameterContainer& parameters,
                                                                                                      const List< String >& parsedMeshType )
{
   std::cerr << "Mesh dimension " << MeshDimensions << " is not supported." << std::endl;
   return false;
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
   template< int MeshDimensions, typename >
bool MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::resolveMeshRealType( const Config::ParameterContainer& parameters,
                                                                                                      const List< String >& parsedMeshType )
{
   if( parsedMeshType[ 2 ] == "float" )
      return resolveMeshIndexType< MeshDimensions, float >( parameters, parsedMeshType );
   if( parsedMeshType[ 2 ] == "double" )
      return resolveMeshIndexType< MeshDimensions, double >( parameters, parsedMeshType );
   if( parsedMeshType[ 2 ] == "long-double" )
      return resolveMeshIndexType< MeshDimensions, long double >( parameters, parsedMeshType );
   std::cerr << "The type '" << parsedMeshType[ 2 ] << "' is not allowed for real type." << std::endl;
   return false;
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
   template< int MeshDimensions,
             typename MeshRealType,
             typename, typename >
bool MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::resolveMeshIndexType( const Config::ParameterContainer& parameters,
                                                                                                        const List< String >& parsedMeshType )
{
   std::cerr << "The type '" << parsedMeshType[ 4 ] << "' is not allowed for real type." << std::endl;
   return false;
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
   template< int MeshDimensions,
             typename MeshRealType,
             typename >
bool MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::resolveMeshIndexType( const Config::ParameterContainer& parameters,
                                                                                                        const List< String >& parsedMeshType )
{
   if( parsedMeshType[ 4 ] == "short int" )
      return resolveMeshType< MeshDimensions, MeshRealType, short int >( parameters, parsedMeshType );
   if( parsedMeshType[ 4 ] == "int" )
      return resolveMeshType< MeshDimensions, MeshRealType, int >( parameters, parsedMeshType );
   if( parsedMeshType[ 4 ] == "long int" )
      return resolveMeshType< MeshDimensions, MeshRealType, long int >( parameters, parsedMeshType );
   std::cerr << "The type '" << parsedMeshType[ 4 ] << "' is not allowed for indexing type." << std::endl;
   return false;
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
   template< int MeshDimensions,
             typename MeshRealType,
             typename MeshIndexType,
             typename, typename >
bool MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::resolveMeshType( const Config::ParameterContainer& parameters,
                                                                                                   const List< String >& parsedMeshType )
{
   std::cerr << "The type '" << parsedMeshType[ 4 ] << "' is not allowed for indexing type." << std::endl;
   return false;
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
   template< int MeshDimensions,
             typename MeshRealType,
             typename MeshIndexType,
             typename >
bool MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::resolveMeshType( const Config::ParameterContainer& parameters,
                                                                                                   const List< String >& parsedMeshType )
{
   if( parsedMeshType[ 0 ] == "Grid" )
   {
      typedef Meshes::Grid< MeshDimensions, MeshRealType, Device, MeshIndexType > MeshType;
      return MeshResolverTerminator< ProblemSetter, Real, Device, Index, MeshType, ConfigTag >::run( parameters );
   }
   std::cerr << "Unknown mesh type " << parsedMeshType[ 0 ] << "." << std::endl;
   return false;
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag >
class MeshResolverTerminator< ProblemSetter, Real, Device, Index, MeshType, ConfigTag, false >
{
   public:
      static bool run( const Config::ParameterContainer& parameters )
      {
         std::cerr << "The mesh type " << TNL::getType< MeshType >() << " is not supported." << std::endl;
         return false;
      };
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag >
class MeshResolverTerminator< ProblemSetter, Real, Device, Index, MeshType, ConfigTag, true >
{
   public:
      static bool run( const Config::ParameterContainer& parameters )
      {
         return ProblemSetter< Real, Device, Index, MeshType, ConfigTag, SolverStarter< ConfigTag > >::run( parameters );
      }
};

} // namespace Solvers
} // namespace TNL
