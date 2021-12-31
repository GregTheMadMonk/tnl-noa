/***************************************************************************
                          MeshTypeResolver.hpp  -  description
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

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
bool
MeshTypeResolver< ConfigTag, Device >::
run( Reader& reader, Functor&& functor )
{
   return detail< Reader, Functor >::resolveCellTopology( reader, std::forward<Functor>(functor) );
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveCellTopology( Reader& reader, Functor&& functor )
{
   switch( reader.getCellShape() )
   {
      case VTK::EntityShape::Line:
         return resolveSpaceDimension< Topologies::Edge >( reader, std::forward<Functor>(functor) );
      case VTK::EntityShape::Triangle:
         return resolveSpaceDimension< Topologies::Triangle >( reader, std::forward<Functor>(functor) );
      case VTK::EntityShape::Quad:
         return resolveSpaceDimension< Topologies::Quadrangle >( reader, std::forward<Functor>(functor) );
      case VTK::EntityShape::Tetra:
         return resolveSpaceDimension< Topologies::Tetrahedron >( reader, std::forward<Functor>(functor) );
      case VTK::EntityShape::Hexahedron:
         return resolveSpaceDimension< Topologies::Hexahedron >( reader, std::forward<Functor>(functor) );
      case VTK::EntityShape::Polygon:
         return resolveSpaceDimension< Topologies::Polygon >( reader, std::forward<Functor>(functor) );
      case VTK::EntityShape::Wedge:
         return resolveSpaceDimension< Topologies::Wedge >( reader, std::forward<Functor>(functor) );
      case VTK::EntityShape::Pyramid:
         return resolveSpaceDimension< Topologies::Pyramid >( reader, std::forward<Functor>(functor) );
      default:
         std::cerr << "unsupported cell topology: " << VTK::getShapeName( reader.getCellShape() ) << std::endl;
         return false;
   }
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< typename CellTopology,
                typename, typename >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveSpaceDimension( Reader& reader, Functor&& functor )
{
   std::cerr << "The cell topology " << getType< CellTopology >() << " is disabled in the build configuration." << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< typename CellTopology,
                typename >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveSpaceDimension( Reader& reader, Functor&& functor )
{
   switch( reader.getSpaceDimension() )
   {
      case 1:
         return resolveReal< CellTopology, 1 >( reader, std::forward<Functor>(functor) );
      case 2:
         return resolveReal< CellTopology, 2 >( reader, std::forward<Functor>(functor) );
      case 3:
         return resolveReal< CellTopology, 3 >( reader, std::forward<Functor>(functor) );
      default:
         std::cerr << "unsupported space dimension: " << reader.getSpaceDimension() << std::endl;
         return false;
   }
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< typename CellTopology,
                int SpaceDimension,
                typename, typename >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveReal( Reader& reader, Functor&& functor )
{
   std::cerr << "The combination of space dimension (" << SpaceDimension
             << ") and mesh dimension (" << CellTopology::dimension
             << ") is either invalid or disabled in the build configuration." << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< typename CellTopology,
                int SpaceDimension,
                typename >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveReal( Reader& reader, Functor&& functor )
{
   if( reader.getRealType() == "float" )
      return resolveGlobalIndex< CellTopology, SpaceDimension, float >( reader, std::forward<Functor>(functor) );
   if( reader.getRealType() == "double" )
      return resolveGlobalIndex< CellTopology, SpaceDimension, double >( reader, std::forward<Functor>(functor) );
   if( reader.getRealType() == "long double" )
      return resolveGlobalIndex< CellTopology, SpaceDimension, long double >( reader, std::forward<Functor>(functor) );
   std::cerr << "Unsupported real type: " << reader.getRealType() << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< typename CellTopology,
                int SpaceDimension,
                typename Real,
                typename, typename >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveGlobalIndex( Reader& reader, Functor&& functor )
{
   std::cerr << "The mesh real type " << getType< Real >() << " is disabled in the build configuration." << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< typename CellTopology,
                int SpaceDimension,
                typename Real,
                typename >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveGlobalIndex( Reader& reader, Functor&& functor )
{
   if( reader.getGlobalIndexType() == "short" ||
       reader.getGlobalIndexType() == "short int" ||
       reader.getGlobalIndexType() == "std::int16_t" ||
       reader.getGlobalIndexType() == "std::uint16_t" )
      return resolveLocalIndex< CellTopology, SpaceDimension, Real, short int >( reader, std::forward<Functor>(functor) );
   if( reader.getGlobalIndexType() == "int" ||
       reader.getGlobalIndexType() == "std::int32_t" ||
       reader.getGlobalIndexType() == "std::uint32_t" )
      return resolveLocalIndex< CellTopology, SpaceDimension, Real, int >( reader, std::forward<Functor>(functor) );
   if( reader.getGlobalIndexType() == "long" ||
       reader.getGlobalIndexType() == "long int" ||
       reader.getGlobalIndexType() == "std::int64_t" ||
       reader.getGlobalIndexType() == "std::uint64_t" )
      return resolveLocalIndex< CellTopology, SpaceDimension, Real, long int >( reader, std::forward<Functor>(functor) );
   std::cerr << "Unsupported global index type: " << reader.getGlobalIndexType() << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< typename CellTopology,
                int SpaceDimension,
                typename Real,
                typename GlobalIndex,
                typename, typename >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveLocalIndex( Reader& reader, Functor&& functor )
{
   std::cerr << "The mesh global index type " << getType< GlobalIndex >() << " is disabled in the build configuration." << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< typename CellTopology,
                int SpaceDimension,
                typename Real,
                typename GlobalIndex,
                typename >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveLocalIndex( Reader& reader, Functor&& functor )
{
   if( reader.getLocalIndexType() == "short" ||
       reader.getLocalIndexType() == "short int" ||
       reader.getLocalIndexType() == "std::int16_t" ||
       reader.getLocalIndexType() == "std::uint16_t" )
      return resolveMeshType< CellTopology, SpaceDimension, Real, GlobalIndex, short int >( reader, std::forward<Functor>(functor) );
   if( reader.getLocalIndexType() == "int" ||
       reader.getLocalIndexType() == "std::int32_t" ||
       reader.getLocalIndexType() == "std::uint32_t" )
      return resolveMeshType< CellTopology, SpaceDimension, Real, GlobalIndex, int >( reader, std::forward<Functor>(functor) );
   if( reader.getLocalIndexType() == "long" ||
       reader.getLocalIndexType() == "long int" ||
       reader.getLocalIndexType() == "std::int64_t" ||
       reader.getLocalIndexType() == "std::uint64_t" )
      return resolveMeshType< CellTopology, SpaceDimension, Real, GlobalIndex, long int >( reader, std::forward<Functor>(functor) );
   std::cerr << "Unsupported local index type: " << reader.getLocalIndexType() << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< typename CellTopology,
                int SpaceDimension,
                typename Real,
                typename GlobalIndex,
                typename LocalIndex,
                typename, typename >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveMeshType( Reader& reader, Functor&& functor )
{
   std::cerr << "The mesh local index type " << getType< LocalIndex >() << " is disabled in the build configuration." << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< typename CellTopology,
                int SpaceDimension,
                typename Real,
                typename GlobalIndex,
                typename LocalIndex,
                typename >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveMeshType( Reader& reader, Functor&& functor )
{
   using MeshConfig = typename BuildConfigTags::MeshConfigTemplateTag< ConfigTag >::template MeshConfig< CellTopology, SpaceDimension, Real, GlobalIndex, LocalIndex >;
   return resolveTerminate< MeshConfig >( reader, std::forward<Functor>(functor) );
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< typename MeshConfig,
                typename, typename >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveTerminate( Reader& reader, Functor&& functor )
{
   std::cerr << "The mesh config type " << getType< MeshConfig >() << " is disabled in the build configuration for device " << getType< Device >() << "." << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< typename MeshConfig,
                typename >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveTerminate( Reader& reader, Functor&& functor )
{
   using MeshType = Meshes::Mesh< MeshConfig, Device >;
   return std::forward<Functor>(functor)( reader, MeshType{} );
}

} // namespace Meshes
} // namespace TNL
