/***************************************************************************
                          EntityShape.h  -  description
                             -------------------
    begin                : Nov 22, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <ostream>
#include <cstdint>

#include <TNL/Meshes/Topologies/Edge.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Quadrilateral.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>
#include <TNL/Meshes/Topologies/Hexahedron.h>

namespace TNL {
namespace Meshes {
namespace Readers {

/*
 * Enumeration of entity shapes, inspired by the VTK library.
 */
enum class EntityShape
: std::uint8_t
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

inline std::ostream& operator<<( std::ostream& str, EntityShape shape )
{
   switch( shape )
   {
      case EntityShape::Vertex:
         str << "Entity::Vertex";
         break;
      case EntityShape::PolyVertex:
         str << "Entity::PolyVertex";
         break;
      case EntityShape::Line:
         str << "Entity::Line";
         break;
      case EntityShape::PolyLine:
         str << "Entity::PolyLine";
         break;
      case EntityShape::Triangle:
         str << "Entity::Triangle";
         break;
      case EntityShape::TriangleStrip:
         str << "Entity::TriangleStrip";
         break;
      case EntityShape::Polygon:
         str << "Entity::Polygon";
         break;
      case EntityShape::Pixel:
         str << "Entity::Pixel";
         break;
      case EntityShape::Quad:
         str << "Entity::Quad";
         break;
      case EntityShape::Tetra:
         str << "Entity::Tetra";
         break;
      case EntityShape::Voxel:
         str << "Entity::Voxel";
         break;
      case EntityShape::Hexahedron:
         str << "Entity::Hexahedron";
         break;
      case EntityShape::Wedge:
         str << "Entity::Wedge";
         break;
      case EntityShape::Pyramid:
         str << "Entity::Pyramid";
         break;
      default:
         str << "<unknown entity>";
   }
   return str;
}

inline int getEntityDimension( EntityShape shape )
{
   switch( shape )
   {
      case EntityShape::Vertex:         return 0;
      case EntityShape::PolyVertex:     return 0;
      case EntityShape::Line:           return 1;
      case EntityShape::PolyLine:       return 1;
      case EntityShape::Triangle:       return 2;
      case EntityShape::TriangleStrip:  return 2;
      case EntityShape::Polygon:        return 2;
      case EntityShape::Pixel:          return 2;
      case EntityShape::Quad:           return 2;
      case EntityShape::Tetra:          return 3;
      case EntityShape::Voxel:          return 3;
      case EntityShape::Hexahedron:     return 3;
      case EntityShape::Wedge:          return 3;
      case EntityShape::Pyramid:        return 3;
   }
   // this just avoids a compiler warning in GCC and nvcc (clang actually knows if the
   // switch above covers all cases, and print a warning only when it does not)
   throw 1;
}

// static mapping of TNL entity topologies to EntityShape
template< typename Topology > struct TopologyToEntityShape {};
template<> struct TopologyToEntityShape< Topologies::Vertex >         { static constexpr EntityShape shape = EntityShape::Vertex; };
template<> struct TopologyToEntityShape< Topologies::Edge >           { static constexpr EntityShape shape = EntityShape::Line; };
template<> struct TopologyToEntityShape< Topologies::Triangle >       { static constexpr EntityShape shape = EntityShape::Triangle; };
template<> struct TopologyToEntityShape< Topologies::Quadrilateral >  { static constexpr EntityShape shape = EntityShape::Quad; };
template<> struct TopologyToEntityShape< Topologies::Tetrahedron >    { static constexpr EntityShape shape = EntityShape::Tetra; };
template<> struct TopologyToEntityShape< Topologies::Hexahedron >     { static constexpr EntityShape shape = EntityShape::Hexahedron; };

// mapping used in VTKWriter
template< typename GridEntity >
struct GridEntityShape
{
private:
   static constexpr int dim = GridEntity::getEntityDimension();
   static_assert( dim >= 0 && dim <= 3, "unexpected dimension of the grid entity" );

public:
   static constexpr EntityShape shape =
      (dim == 0) ? EntityShape::Vertex :
      (dim == 1) ? EntityShape::Line :
      (dim == 2) ? EntityShape::Pixel :
                   EntityShape::Voxel;
};

} // namespace Readers
} // namespace Meshes
} // namespace TNL
