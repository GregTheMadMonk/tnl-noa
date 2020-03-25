/***************************************************************************
                          MeshTypeResolver_impl.h  -  description
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
#include <TNL/Meshes/TypeResolver/MeshTypeResolver.h>
#include <TNL/Meshes/VTKTraits.h>

namespace TNL {
namespace Meshes {

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
bool
MeshTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
run( const Reader& reader,
     ProblemSetterArgs&&... problemSetterArgs )
{
   return resolveCellTopology( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
bool
MeshTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveCellTopology( const Reader& reader,
                     ProblemSetterArgs&&... problemSetterArgs )
{
   switch( reader.getCellShape() )
   {
      case VTK::EntityShape::Line:
         return resolveWorldDimension< Topologies::Edge >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
      case VTK::EntityShape::Triangle:
         return resolveWorldDimension< Topologies::Triangle >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
      case VTK::EntityShape::Quad:
         return resolveWorldDimension< Topologies::Quadrilateral >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
      case VTK::EntityShape::Tetra:
         return resolveWorldDimension< Topologies::Tetrahedron >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
      case VTK::EntityShape::Hexahedron:
         return resolveWorldDimension< Topologies::Hexahedron >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
      default:
         std::cerr << "unsupported cell topology: " << VTK::getShapeName( reader.getCellShape() ) << std::endl;
         return false;
   }
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< typename CellTopology,
             typename, typename >
bool
MeshTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveWorldDimension( const Reader& reader,
                       ProblemSetterArgs&&... problemSetterArgs )
{
   std::cerr << "The cell topology " << getType< CellTopology >() << " is disabled in the build configuration." << std::endl;
   return false;
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< typename CellTopology,
             typename >
bool
MeshTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveWorldDimension( const Reader& reader,
                       ProblemSetterArgs&&... problemSetterArgs )
{
   switch( reader.getWorldDimension() )
   {
      case 1:
         return resolveReal< CellTopology, 1 >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
      case 2:
         return resolveReal< CellTopology, 2 >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
      case 3:
         return resolveReal< CellTopology, 3 >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
      default:
         std::cerr << "unsupported world dimension: " << reader.getWorldDimension() << std::endl;
         return false;
   }
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< typename CellTopology,
             int WorldDimension,
             typename, typename >
bool
MeshTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveReal( const Reader& reader,
             ProblemSetterArgs&&... problemSetterArgs )
{
   std::cerr << "The combination of world dimension (" << WorldDimension
             << ") and mesh dimension (" << CellTopology::dimension
             << ") is either invalid or disabled in the build configuration." << std::endl;
   return false;
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< typename CellTopology,
             int WorldDimension,
             typename >
bool
MeshTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveReal( const Reader& reader,
             ProblemSetterArgs&&... problemSetterArgs )
{
   if( reader.getRealType() == "float" )
      return resolveGlobalIndex< CellTopology, WorldDimension, float >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   if( reader.getRealType() == "double" )
      return resolveGlobalIndex< CellTopology, WorldDimension, double >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   if( reader.getRealType() == "long double" )
      return resolveGlobalIndex< CellTopology, WorldDimension, long double >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   std::cerr << "Unsupported real type: " << reader.getRealType() << std::endl;
   return false;
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< typename CellTopology,
             int WorldDimension,
             typename Real,
             typename, typename >
bool
MeshTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveGlobalIndex( const Reader& reader,
                    ProblemSetterArgs&&... problemSetterArgs )
{
   std::cerr << "The mesh real type " << getType< Real >() << " is disabled in the build configuration." << std::endl;
   return false;
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< typename CellTopology,
             int WorldDimension,
             typename Real,
             typename >
bool
MeshTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveGlobalIndex( const Reader& reader,
                    ProblemSetterArgs&&... problemSetterArgs )
{
   if( reader.getGlobalIndexType() == "short int" ||
       reader.getGlobalIndexType() == "std::int16_t" ||
       reader.getGlobalIndexType() == "std::uint16_t" )
      return resolveLocalIndex< CellTopology, WorldDimension, Real, short int >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   if( reader.getGlobalIndexType() == "int" ||
       reader.getGlobalIndexType() == "std::int32_t" ||
       reader.getGlobalIndexType() == "std::uint32_t" )
      return resolveLocalIndex< CellTopology, WorldDimension, Real, int >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   if( reader.getGlobalIndexType() == "long int" ||
       reader.getGlobalIndexType() == "std::int64_t" ||
       reader.getGlobalIndexType() == "std::uint64_t" )
      return resolveLocalIndex< CellTopology, WorldDimension, Real, long int >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   std::cerr << "Unsupported global index type: " << reader.getGlobalIndexType() << std::endl;
   return false;
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< typename CellTopology,
             int WorldDimension,
             typename Real,
             typename GlobalIndex,
             typename, typename >
bool
MeshTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveLocalIndex( const Reader& reader,
                   ProblemSetterArgs&&... problemSetterArgs )
{
   std::cerr << "The mesh global index type " << getType< GlobalIndex >() << " is disabled in the build configuration." << std::endl;
   return false;
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< typename CellTopology,
             int WorldDimension,
             typename Real,
             typename GlobalIndex,
             typename >
bool
MeshTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveLocalIndex( const Reader& reader,
                   ProblemSetterArgs&&... problemSetterArgs )
{
   if( reader.getLocalIndexType() == "short int" ||
       reader.getLocalIndexType() == "std::int16_t" ||
       reader.getLocalIndexType() == "std::uint16_t" )
      return resolveMeshType< CellTopology, WorldDimension, Real, GlobalIndex, short int >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   if( reader.getLocalIndexType() == "int" ||
       reader.getLocalIndexType() == "std::int32_t" ||
       reader.getLocalIndexType() == "std::uint32_t" )
      return resolveMeshType< CellTopology, WorldDimension, Real, GlobalIndex, int >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   if( reader.getLocalIndexType() == "long int" ||
       reader.getLocalIndexType() == "std::int64_t" ||
       reader.getLocalIndexType() == "std::uint64_t" )
      return resolveMeshType< CellTopology, WorldDimension, Real, GlobalIndex, long int >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
   std::cerr << "Unsupported local index type: " << reader.getLocalIndexType() << std::endl;
   return false;
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< typename CellTopology,
             int WorldDimension,
             typename Real,
             typename GlobalIndex,
             typename LocalIndex,
             typename, typename >
bool
MeshTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveMeshType( const Reader& reader,
           ProblemSetterArgs&&... problemSetterArgs )
{
   std::cerr << "The mesh local index type " << getType< LocalIndex >() << " is disabled in the build configuration." << std::endl;
   return false;
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< typename CellTopology,
             int WorldDimension,
             typename Real,
             typename GlobalIndex,
             typename LocalIndex,
             typename >
bool
MeshTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveMeshType( const Reader& reader,
                 ProblemSetterArgs&&... problemSetterArgs )
{
   using MeshConfig = typename BuildConfigTags::MeshConfigTemplateTag< ConfigTag >::template MeshConfig< CellTopology, WorldDimension, Real, GlobalIndex, LocalIndex >;
   return resolveTerminate< MeshConfig >( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< typename MeshConfig,
             typename, typename >
bool
MeshTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveTerminate( const Reader& reader,
                  ProblemSetterArgs&&... problemSetterArgs )
{
   std::cerr << "The mesh config type " << getType< MeshConfig >() << " is disabled in the build configuration for device " << getType< Device >() << "." << std::endl;
   return false;
}

template< typename Reader,
          typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
   template< typename MeshConfig,
             typename >
bool
MeshTypeResolver< Reader, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
resolveTerminate( const Reader& reader,
                  ProblemSetterArgs&&... problemSetterArgs )
{
   using MeshType = Meshes::Mesh< MeshConfig, Device >;
   return ProblemSetter< MeshType >::run( std::forward<ProblemSetterArgs>(problemSetterArgs)... );
}

} // namespace Meshes
} // namespace TNL
