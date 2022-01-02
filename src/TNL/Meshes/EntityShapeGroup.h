#pragma once

#include <TNL/Meshes/VTKTraits.h>

namespace TNL {
namespace Meshes {
namespace VTK {

template < EntityShape GeneralShape >
struct EntityShapeGroup
{
};

template < EntityShape GeneralShape, int index >
struct EntityShapeGroupElement
{ 
};

template <>
struct EntityShapeGroup< EntityShape::Polygon >
{
   static constexpr int size = 2;
};

template <>
struct EntityShapeGroupElement< EntityShape::Polygon, 0 >
{
   static constexpr EntityShape shape = EntityShape::Triangle;
};

template <>
struct EntityShapeGroupElement< EntityShape::Polygon, 1 >
{
   static constexpr EntityShape shape = EntityShape::Quad;
};

} // namespace VTK
} // namespace Meshes
} // namespace TNL