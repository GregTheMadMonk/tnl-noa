/***************************************************************************
                          GridTypeResolver_impl.h  -  description
                             -------------------
    begin                : Nov 22, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <utility>

#include <TNL/String.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/TypeResolver/GridTypeResolver.h>

namespace TNL {
namespace Meshes {   

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
bool
GridTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
run( const Reader& reader,
     ProblemSetterArgs&&... problemSetterArgs )
{
   return resolveGridDimension( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
bool
GridTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveGridDimension( const Reader& reader,
                      ProblemSetterArgs&&... problemSetterArgs )
{
   if( reader.getMeshDimension() == 1 )
      return resolveReal< 1 >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   if( reader.getMeshDimension() == 2 )
      return resolveReal< 2 >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   if( reader.getMeshDimension() == 3 )
      return resolveReal< 3 >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   std::cerr << "Unsupported mesh dimension: " << reader.getMeshDimension() << std::endl;
   return false;
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< int MeshDimension,
             typename, typename >
bool
GridTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveReal( const Reader& reader,
             ProblemSetterArgs&&... problemSetterArgs )
{
   std::cerr << "The grid dimension " << MeshDimension << " is disabled in the build configuration." << std::endl;
   return false;
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< int MeshDimension,
             typename >
bool
GridTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveReal( const Reader& reader,
             ProblemSetterArgs&&... problemSetterArgs )
{
   if( reader.getRealType() == "float" )
      return resolveIndex< MeshDimension, float >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   if( reader.getRealType() == "double" )
      return resolveIndex< MeshDimension, double >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   if( reader.getRealType() == "long-double" )
      return resolveIndex< MeshDimension, long double >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   std::cerr << "Unsupported real type: " << reader.getRealType() << std::endl;
   return false;
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< int MeshDimension,
             typename Real,
             typename, typename >
bool
GridTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveIndex( const Reader& reader,
              ProblemSetterArgs&&... problemSetterArgs )
{
   std::cerr << "The grid real type " << getType< Real >() << " is disabled in the build configuration." << std::endl;
   return false;
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< int MeshDimension,
             typename Real,
             typename >
bool
GridTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveIndex( const Reader& reader,
              ProblemSetterArgs&&... problemSetterArgs )
{
   if( reader.getGlobalIndexType() == "short int" )
      return resolveGridType< MeshDimension, Real, short int >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   if( reader.getGlobalIndexType() == "int" )
      return resolveGridType< MeshDimension, Real, int >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   if( reader.getGlobalIndexType() == "long int" )
      return resolveGridType< MeshDimension, Real, long int >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   std::cerr << "Unsupported index type: " << reader.getRealType() << std::endl;
   return false;
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< int MeshDimension,
             typename Real,
             typename Index,
             typename, typename >
bool
GridTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveGridType( const Reader& reader,
                 ProblemSetterArgs&&... problemSetterArgs )
{
   std::cerr << "The grid index type " << getType< Index >() << " is disabled in the build configuration." << std::endl;
   return false;
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< int MeshDimension,
             typename Real,
             typename Index,
             typename >
bool
GridTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveGridType( const Reader& reader,
                 ProblemSetterArgs&&... problemSetterArgs )
{
   using GridType = Meshes::Grid< MeshDimension, Real, Device, Index >;
   return resolveTerminate< GridType >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< typename GridType,
             typename, typename >
bool
GridTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveTerminate( const Reader& reader,
                  ProblemSetterArgs&&... problemSetterArgs )
{
   std::cerr << "The mesh type " << TNL::getType< GridType >() << " is disabled in the build configuration." << std::endl;
   return false;
};

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< typename GridType,
             typename >
bool
GridTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveTerminate( const Reader& reader,
                  ProblemSetterArgs&&... problemSetterArgs )
{
   return ProblemSetter< GridType >::run( std::forward<ProblemSetterArgs>(problemSetterArgs)... );
};

} // namespace Meshes
} // namespace TNL
