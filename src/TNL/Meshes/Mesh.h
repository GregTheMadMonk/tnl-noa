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
      using CellType        = typename MeshTraitsType::CellType;
      using VertexType      = typename MeshTraitsType::VertexType;
      using PointType       = typename MeshTraitsType::PointType;
      static constexpr int dimensions = MeshTraitsType::meshDimensions;

      template< int Dimensions >
      using EntityTraits = typename MeshTraitsType::template EntityTraits< Dimensions >;

      template< int Dimensions >
      using EntityType = typename EntityTraits< Dimensions >::EntityType;

      static String getType();

      virtual String getTypeVirtual() const;

      static constexpr int getDimensions();

      template< int Dimensions >
      static constexpr bool entitiesAvailable();

      GlobalIndexType getNumberOfCells() const;

      // TODO: rename to getEntitiesCount
      template< int Dimension >
      GlobalIndexType getNumberOfEntities() const;

      CellType& getCell( const GlobalIndexType entityIndex );

      const CellType& getCell( const GlobalIndexType entityIndex ) const;

      template< int Dimensions >
      EntityType< Dimensions >& getEntity( const GlobalIndexType entityIndex );

      template< int Dimensions >
      const EntityType< Dimensions >& getEntity( const GlobalIndexType entityIndex ) const;

      bool save( File& file ) const;

      bool load( File& file );

      using Object::load;
      using Object::save;

      void print( std::ostream& str ) const;

      bool operator==( const Mesh& mesh ) const;

      bool init( const typename MeshTraitsType::PointArrayType& points,
                 const typename MeshTraitsType::CellSeedArrayType& cellSeeds );

   protected:
      // Methods for the mesh initializer
      using StorageBaseType::getSubentityStorageNetwork;
      using StorageBaseType::getSuperentityStorageNetwork;

      MeshConfigValidator< MeshConfig > configValidator;

      friend MeshInitializer< MeshConfig >;

      template< typename Mesh, typename DimensionsTag, typename SuperdimensionsTag >
      friend struct MeshEntityStorageRebinderWorker;
};

template< typename MeshConfig >
std::ostream& operator<<( std::ostream& str, const Mesh< MeshConfig >& mesh );

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/MeshDetails/Mesh_impl.h>
