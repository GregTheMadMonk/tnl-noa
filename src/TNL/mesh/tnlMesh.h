/***************************************************************************
                          tnlMesh.h  -  description
                             -------------------
    begin                : Feb 16, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <ostream>
#include <TNL/tnlObject.h>
#include <TNL/mesh/tnlMeshEntity.h>
#include <TNL/mesh/traits/tnlMeshTraits.h>
#include <TNL/mesh/layers/tnlMeshStorageLayer.h>
#include <TNL/mesh/config/tnlMeshConfigValidator.h>
#include <TNL/mesh/initializer/tnlMeshInitializer.h>

namespace TNL {

template< typename MeshConfig > //,
          //typename Device = tnlHost >
class tnlMesh : public tnlObject/*,
                public tnlMeshStorageLayers< MeshConfig >*/
{
   public:
 
      typedef MeshConfig                                        Config;
      typedef tnlMeshTraits< MeshConfig >                       MeshTraits;
      typedef typename MeshTraits::DeviceType                   DeviceType;
      typedef typename MeshTraits::GlobalIndexType              GlobalIndexType;
      typedef typename MeshTraits::LocalIndexType               LocalIndexType;
      typedef typename MeshTraits::CellType                     CellType;
      typedef typename MeshTraits::VertexType                   VertexType;
      typedef typename MeshTraits::PointType                    PointType;
      static const int dimensions = MeshTraits::meshDimensions;
      template< int Dimensions > using EntityTraits = typename MeshTraits::template EntityTraits< Dimensions >;
      template< int Dimensions > using EntityType = typename EntityTraits< Dimensions >::EntityType;

      static tnlString getType();
 
      virtual tnlString getTypeVirtual() const;
 
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

      bool save( tnlFile& file ) const;

      bool load( tnlFile& file );
 
      using tnlObject::load;
      using tnlObject::save;
 
      void print( std::ostream& str ) const;

      bool operator==( const tnlMesh& mesh ) const;

      // TODO: this is only for mesh intializer - remove it if possible
      template< typename DimensionsTag >
           typename EntityTraits< DimensionsTag::value >::StorageArrayType& entitiesArray();

 
      template< typename DimensionsTag, typename SuperDimensionsTag >
           typename tnlMeshTraits< MeshConfig >::GlobalIdArrayType& superentityIdsArray();
 
      template< typename EntityTopology, typename SuperdimensionsTag >
      typename MeshTraits::template SuperentityTraits< EntityTopology, SuperdimensionsTag::value >::StorageNetworkType&
      getSuperentityStorageNetwork()
      {
         return entitiesStorage.template getSuperentityStorageNetwork< SuperdimensionsTag >( tnlDimensionsTag< EntityTopology::dimensions >() );
      }
 
      bool init( const typename MeshTraits::PointArrayType& points,
                 const typename MeshTraits::CellSeedArrayType& cellSeeds );
 
 
   protected:
 
      tnlMeshStorageLayers< MeshConfig > entitiesStorage;
 
      tnlMeshConfigValidator< MeshConfig > configValidator;
};

template< typename MeshConfig >
std::ostream& operator <<( std::ostream& str, const tnlMesh< MeshConfig >& mesh );

} // namespace TNL

#include <TNL/mesh/tnlMesh_impl.h>
