/***************************************************************************
                          Mesh.h  -  description
                             -------------------
    begin                : Feb 16, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <ostream>
#include <TNL/Object.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/MeshDetails/layers/MeshStorageLayer.h>
#include <TNL/Meshes/MeshDetails/config/MeshConfigValidator.h>
#include <TNL/Meshes/MeshDetails/initializer/MeshInitializer.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig > //,
          //typename Device = Devices::Host >
class Mesh : public Object/*,
                public MeshStorageLayers< MeshConfig >*/
{
   public:
 
      typedef MeshConfig                                        Config;
      typedef MeshTraits< MeshConfig >                          MeshTraitsType;
      typedef typename MeshTraitsType::DeviceType               DeviceType;
      typedef typename MeshTraitsType::GlobalIndexType          GlobalIndexType;
      typedef typename MeshTraitsType::LocalIndexType           LocalIndexType;
      typedef typename MeshTraitsType::CellType                 CellType;
      typedef typename MeshTraitsType::VertexType               VertexType;
      typedef typename MeshTraitsType::PointType                PointType;
      static const int dimensions = MeshTraitsType::meshDimensions;
      template< int Dimensions > using EntityTraits = typename MeshTraitsType::template EntityTraits< Dimensions >;
      template< int Dimensions > using EntityType = typename EntityTraits< Dimensions >::EntityType;

      static String getType();
 
      virtual String getTypeVirtual() const;
 
      static constexpr int getDimensions();

      template< int Dimensions >
      bool entitiesAvalable() const;
 
      GlobalIndexType getNumberOfCells() const;

      // TODO: rename to getEntitiesCount
      template< int Dimensions >
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

      // TODO: this is only for mesh intializer - remove it if possible
      template< typename DimensionsTag >
           typename EntityTraits< DimensionsTag::value >::StorageArrayType& entitiesArray();

 
      template< typename DimensionsTag, typename SuperDimensionsTag >
           typename MeshTraits< MeshConfig >::GlobalIdArrayType& superentityIdsArray();
 
      template< typename EntityTopology, typename SuperdimensionsTag >
      typename MeshTraitsType::template SuperentityTraits< EntityTopology, SuperdimensionsTag::value >::StorageNetworkType&
      getSuperentityStorageNetwork()
      {
         return entitiesStorage.template getSuperentityStorageNetwork< SuperdimensionsTag >( MeshDimensionsTag< EntityTopology::dimensions >() );
      }
 
      bool init( const typename MeshTraitsType::PointArrayType& points,
                 const typename MeshTraitsType::CellSeedArrayType& cellSeeds );
 
 
   protected:
 
      MeshStorageLayers< MeshConfig > entitiesStorage;
 
      MeshConfigValidator< MeshConfig > configValidator;
};

template< typename MeshConfig >
std::ostream& operator <<( std::ostream& str, const Mesh< MeshConfig >& mesh );

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/MeshDetails/Mesh_impl.h>
