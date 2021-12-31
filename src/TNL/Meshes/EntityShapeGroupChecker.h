#pragma once

#include <TNL/Algorithms/TemplateStaticFor.h>
#include <TNL/Meshes/EntityShapeGroup.h>

namespace TNL {
namespace Meshes {
namespace VTK {

template< EntityShape GeneralShape_ >
class EntityShapeGroupChecker
{
public:
   static constexpr EntityShape GeneralShape = GeneralShape_;

   static bool belong( EntityShape shape )
   {
      if( GeneralShape == shape )
      {
         return true;
      }
      else
      {
         bool result = false;
         Algorithms::TemplateStaticFor< int, 0, EntityShapeGroup< GeneralShape >::size, OtherEntitiesChecker >::execHost( result, shape );
         return result;
      }
   }

   static bool bothBelong( EntityShape shape1, EntityShape shape2 )
   {
      return belong( shape1 ) && belong( shape2 );
   }

private:
   template< int index >
   class OtherEntitiesChecker
   {
      public:
         static void exec( bool& result, EntityShape shape )
         {
            EntityShape groupShape = EntityShapeGroupElement< GeneralShape, index >::shape;
            result = result || ( shape == groupShape );
         }
   };
};

} // namespace VTK
} // namespace Meshes
} // namespace TNL