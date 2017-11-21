/***************************************************************************
                          MeshTypeResolver_impl.h  -  description
                             -------------------
    begin                : Nov 22, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <ostream>

#include <TNL/Meshes/Topologies/Edge.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Quadrilateral.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>
#include <TNL/Meshes/Topologies/Hexahedron.h>

namespace TNL {
namespace Meshes {
namespace Readers {

/*
 * Entity types used in the VTK library. For convenience, we use them also in
 * our Mesh readers.
 */
enum class VTKEntityType
{
   Vertex = 1,
   PolyVertex = 2,
   Line = 3,
   PolyLine = 4,
   Triangle = 5,
   TriangleStrip = 6,
   Polygon = 7,
   Pixel = 8,
   Quad = 9,
   Tetra = 10,
   Voxel = 11,
   Hexahedron = 12,
   Wedge = 13,
   Pyramid = 14
};

static std::ostream& operator<<( std::ostream& str, VTKEntityType type )
{
   switch( type )
   {
      case VTKEntityType::Vertex:
         str << "VTK::Vertex";
         break;
      case VTKEntityType::PolyVertex:
         str << "VTK::PolyVertex";
         break;
      case VTKEntityType::Line:
         str << "VTK::Line";
         break;
      case VTKEntityType::PolyLine:
         str << "VTK::PolyLine";
         break;
      case VTKEntityType::Triangle:
         str << "VTK::Triangle";
         break;
      case VTKEntityType::TriangleStrip:
         str << "VTK::TriangleStrip";
         break;
      case VTKEntityType::Polygon:
         str << "VTK::Polygon";
         break;
      case VTKEntityType::Pixel:
         str << "VTK::Pixel";
         break;
      case VTKEntityType::Quad:
         str << "VTK::Quad";
         break;
      case VTKEntityType::Tetra:
         str << "VTK::Tetra";
         break;
      case VTKEntityType::Voxel:
         str << "VTK::Voxel";
         break;
      case VTKEntityType::Hexahedron:
         str << "VTK::Hexahedron";
         break;
      case VTKEntityType::Wedge:
         str << "VTK::Wedge";
         break;
      case VTKEntityType::Pyramid:
         str << "VTK::Pyramid";
         break;
      default:
         str << "<unknown entity>";
   }
   return str;
}

static int getVTKEntityDimension( VTKEntityType type )
{
   switch( type )
   {
      case VTKEntityType::Vertex:         return 0;
      case VTKEntityType::PolyVertex:     return 0;
      case VTKEntityType::Line:           return 1;
      case VTKEntityType::PolyLine:       return 1;
      case VTKEntityType::Triangle:       return 2;
      case VTKEntityType::TriangleStrip:  return 2;
      case VTKEntityType::Polygon:        return 2;
      case VTKEntityType::Pixel:          return 2;
      case VTKEntityType::Quad:           return 2;
      case VTKEntityType::Tetra:          return 3;
      case VTKEntityType::Voxel:          return 3;
      case VTKEntityType::Hexahedron:     return 3;
      case VTKEntityType::Wedge:          return 3;
      case VTKEntityType::Pyramid:        return 3;
   }
   // this just avoids a compiler warning in GCC and nvcc (clang actually knows if the
   // switch above covers all cases, and print a warning only when it does not)
   throw 1;
}

// static mapping of TNL entity topologies to VTK types
template< typename Topology > struct TopologyToVTKMap {};
template<> struct TopologyToVTKMap< Meshes::Topologies::Vertex >         { static constexpr VTKEntityType type = VTKEntityType::Vertex; };
template<> struct TopologyToVTKMap< Meshes::Topologies::Edge >           { static constexpr VTKEntityType type = VTKEntityType::Line; };
template<> struct TopologyToVTKMap< Meshes::Topologies::Triangle >       { static constexpr VTKEntityType type = VTKEntityType::Triangle; };
template<> struct TopologyToVTKMap< Meshes::Topologies::Quadrilateral >  { static constexpr VTKEntityType type = VTKEntityType::Quad; };
template<> struct TopologyToVTKMap< Meshes::Topologies::Tetrahedron >    { static constexpr VTKEntityType type = VTKEntityType::Tetra; };
template<> struct TopologyToVTKMap< Meshes::Topologies::Hexahedron >     { static constexpr VTKEntityType type = VTKEntityType::Hexahedron; };

} // namespace Readers
} // namespace Meshes
} // namespace TNL
