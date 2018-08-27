/***************************************************************************
                          MeshFunctionGnuplotWriter.h  -  description
                             -------------------
    begin                : Jan 28, 2016
    copyright            : (C) 2016 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>

namespace TNL {

namespace Meshes {
   template< typename, typename, typename > class MeshEntity;
}

namespace Functions {

template< typename MeshFunction >
class MeshFunctionGnuplotWriter
{
   using MeshType = typename MeshFunction::MeshType;
   using EntityType = typename MeshType::template EntityType< MeshFunction::getEntitiesDimension() >;
   using GlobalIndex = typename MeshType::GlobalIndexType;

   template< typename Entity, int dim = Entity::getEntityDimension() >
   struct center
   {
      static auto get( const Entity& entity ) -> decltype(entity.getCenter())
      {
         return entity.getCenter();
      }
   };

   template< typename Entity >
   struct center< Entity, 0 >
   {
      static auto get( const Entity& entity ) -> decltype(entity.getPoint())
      {
         return entity.getPoint();
      }
   };

   template< typename MeshConfig, typename Device, typename Topology, int dim >
   struct center< TNL::Meshes::MeshEntity< MeshConfig, Device, Topology >, dim >
   {
      static int get( const TNL::Meshes::MeshEntity< MeshConfig, Device, Topology >& entity )
      {
         throw "not implemented";
      }
   };

   template< typename MeshConfig, typename Device, typename Topology >
   struct center< TNL::Meshes::MeshEntity< MeshConfig, Device, Topology >, 0 >
   {
      static int get( const TNL::Meshes::MeshEntity< MeshConfig, Device, Topology >& entity )
      {
         throw "not implemented";
      }
   };

public:
   static bool write( const MeshFunction& function,
                      std::ostream& str,
                      const double& scale = 1.0 )
   {
      const MeshType& mesh = function.getMesh();
      const GlobalIndex entitiesCount = mesh.template getEntitiesCount< EntityType >();
      for( GlobalIndex i = 0; i < entitiesCount; i++ ) {
         const EntityType& entity = mesh.template getEntity< EntityType >( i );
         typename MeshType::PointType v = center< EntityType >::get( entity );
         for( int j = 0; j < v.getSize(); j++ )
            str << v[ j ] << " ";
         str << scale * function.getData().getElement( i ) << "\n";
      }
      return true;
   }
};

} // namespace Functions
} // namespace TNL
