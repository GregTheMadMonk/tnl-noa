/***************************************************************************
                          MeshTypeResolver_impl.h  -  description
                             -------------------
    begin                : Nov 28, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <utility>

#include <TNL/String.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Solvers/MeshTypeResolver.h>

namespace TNL {
namespace Solvers {   

template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
bool
MeshTypeResolver< ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
run( const String& fileName,
     ProblemSetterArgs&&... problemSetterArgs )
{
   Meshes::Readers::TNL reader;
   if( ! reader.readFile( fileName ) )
      return false;
   return resolveMeshDimension( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
}

template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
bool
MeshTypeResolver< ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveMeshDimension( Meshes::Readers::TNL& reader,
                      ProblemSetterArgs&&... problemSetterArgs )
{
   if( reader.getMeshDimension() == 1 )
      return resolveMeshRealType< 1 >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   if( reader.getMeshDimension() == 2 )
      return resolveMeshRealType< 2 >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   if( reader.getMeshDimension() == 3 )
      return resolveMeshRealType< 3 >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   std::cerr << "Unsupported mesh dimension: " << reader.getMeshDimension() << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< int MeshDimension >
bool
MeshTypeResolver< ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveMeshRealType( Meshes::Readers::TNL& reader,
                     ProblemSetterArgs&&... problemSetterArgs )
{
   if( reader.getRealType() == "float" )
      return resolveMeshIndexType< MeshDimension, float >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   if( reader.getRealType() == "double" )
      return resolveMeshIndexType< MeshDimension, double >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   if( reader.getRealType() == "long-double" )
      return resolveMeshIndexType< MeshDimension, long double >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   std::cerr << "Unsupported real type: " << reader.getRealType() << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< int MeshDimension,
             typename MeshRealType >
bool
MeshTypeResolver< ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveMeshIndexType( Meshes::Readers::TNL& reader,
                      ProblemSetterArgs&&... problemSetterArgs )
{
   if( reader.getIndexType() == "short int" )
      return resolveMeshType< MeshDimension, MeshRealType, short int >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   if( reader.getIndexType() == "int" )
      return resolveMeshType< MeshDimension, MeshRealType, int >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   if( reader.getIndexType() == "long int" )
      return resolveMeshType< MeshDimension, MeshRealType, long int >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   std::cerr << "Unsupported index type: " << reader.getRealType() << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< int MeshDimension,
             typename MeshRealType,
             typename MeshIndexType >
bool
MeshTypeResolver< ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveMeshType( Meshes::Readers::TNL& reader,
                 ProblemSetterArgs&&... problemSetterArgs )
{
   using MeshType = Meshes::Grid< MeshDimension, MeshRealType, Device, MeshIndexType >;
   return resolveTerminate< MeshType >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
}

template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< typename MeshType,
             typename, typename >
bool
MeshTypeResolver< ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveTerminate( Meshes::Readers::TNL& reader,
                  ProblemSetterArgs&&... problemSetterArgs )
{
   std::cerr << "The mesh type " << TNL::getType< MeshType >() << " is disabled in the build configuration." << std::endl;
   return false;
};

template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< typename MeshType,
             typename >
bool
MeshTypeResolver< ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveTerminate( Meshes::Readers::TNL& reader,
                  ProblemSetterArgs&&... problemSetterArgs )
{
   // TODO: the mesh can be loaded here (this will be probably necessary for unstructured meshes anyway)
   return ProblemSetter< MeshType >::run( std::forward<ProblemSetterArgs>(problemSetterArgs)... );
};

} // namespace Solvers
} // namespace TNL
