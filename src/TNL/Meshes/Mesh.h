/***************************************************************************
                          Mesh.h  -  description
                             -------------------
    begin                : Feb 16, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <ostream>
#include <TNL/Object.h>
#include <TNL/Logger.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/MeshDetails/layers/MeshStorageLayer.h>
#include <TNL/Meshes/MeshDetails/config/MeshConfigValidator.h>
#include <TNL/Meshes/MeshDetails/initializer/MeshInitializer.h>
#include <TNL/Meshes/MeshDetails/layers/MeshEntityStorageRebinder.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig > //,
          //typename Device = Devices::Host >
class Mesh
   : public Object,
     protected MeshStorageLayers< MeshConfig >
{
      using StorageBaseType = MeshStorageLayers< MeshConfig >;

   public:
      using Config          = MeshConfig;
      using MeshTraitsType  = MeshTraits< MeshConfig >;
      using DeviceType      = typename MeshTraitsType::DeviceType;
      using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
      using LocalIndexType  = typename MeshTraitsType::LocalIndexType;
      using PointType       = typename MeshTraitsType::PointType;
      using RealType        = typename PointType::RealType;

      template< int Dimension >
      using EntityTraits = typename MeshTraitsType::template EntityTraits< Dimension >;

      template< int Dimension >
      using EntityType = typename EntityTraits< Dimension >::EntityType;

      static constexpr int getMeshDimension();

      // types of common entities
      using Cell = EntityType< getMeshDimension() >;
      using Face = EntityType< getMeshDimension() - 1 >;
      using Vertex = EntityType< 0 >;

      static String getType();

      virtual String getTypeVirtual() const;

      static String getSerializationType();

      virtual String getSerializationTypeVirtual() const;

      using StorageBaseType::isBoundaryEntity;
      using StorageBaseType::getBoundaryEntitiesCount;
      using StorageBaseType::getBoundaryEntityIndex;
      using StorageBaseType::getInteriorEntitiesCount;
      using StorageBaseType::getInteriorEntityIndex;

      template< int Dimension >
      static constexpr bool entitiesAvailable();

      template< int Dimension >
      GlobalIndexType getEntitiesCount() const;

      template< int Dimension >
      EntityType< Dimension >& getEntity( const GlobalIndexType& entityIndex );

      template< int Dimension >
      const EntityType< Dimension >& getEntity( const GlobalIndexType& entityIndex ) const;


      // duplicated for compatibility with grids
      template< typename EntityType >
      GlobalIndexType getEntitiesCount() const;

      template< typename EntityType >
      EntityType& getEntity( const GlobalIndexType& entityIndex );

      template< typename EntityType >
      const EntityType& getEntity( const GlobalIndexType& entityIndex ) const;


      bool save( File& file ) const;

      bool load( File& file );

      using Object::load;
      using Object::save;

      void print( std::ostream& str ) const;

      bool operator==( const Mesh& mesh ) const;

      // The points and cellSeeds arrays will be reset when not needed to save memory.
      bool init( typename MeshTraitsType::PointArrayType& points,
                 typename MeshTraitsType::CellSeedArrayType& cellSeeds );

      void writeProlog( Logger& logger );

   protected:
      // Methods for the mesh initializer
      using StorageBaseType::setNumberOfEntities;
      using StorageBaseType::getSubentityStorageNetwork;
      using StorageBaseType::getSuperentityStorageNetwork;

      MeshConfigValidator< MeshConfig > configValidator;

      friend MeshInitializer< MeshConfig >;

      template< typename Mesh, typename DimensionTag, typename SuperdimensionTag >
      friend struct MeshEntityStorageRebinderWorker;
};

template< typename MeshConfig >
std::ostream& operator<<( std::ostream& str, const Mesh< MeshConfig >& mesh );

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/MeshDetails/Mesh_impl.h>
