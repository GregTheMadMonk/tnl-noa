/***************************************************************************
                          tnlMesh.h  -  description
                             -------------------
    begin                : Feb 16, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLMESH_H_
#define TNLMESH_H_

#include <core/tnlObject.h>
#include <mesh/tnlMeshEntity.h>
#include <mesh/traits/tnlMeshTraits.h>
#include <mesh/layers/tnlMeshStorageLayer.h>
#include <mesh/config/tnlMeshConfigValidator.h>
#include <mesh/tnlMeshInitializer.h>

template< typename MeshConfig >
class tnlMesh : public tnlObject,
                public tnlMeshStorageLayers< MeshConfig >
{
   public:
   typedef MeshConfig                                        Config;
   typedef tnlMeshTraits< MeshConfig >                       MeshTraits;
   typedef typename tnlMeshTraits< MeshConfig >::PointType   PointType;
   static const int dimensions = MeshTraits::meshDimensions;
   template< typename Dimensions > using EntityTraits = typename MeshTraits::template EntityTraits< Dimensions::value >;

   static tnlString getType()
   {
      return tnlString( "tnlMesh< ") + MeshConfig::getType() + " >";
   }

   virtual tnlString getTypeVirtual() const
   {
      return this->getType();
   }

   using tnlObject::save;
   using tnlObject::load;

   bool save( tnlFile& file ) const
   {
      if( ! tnlObject::save( file ) ||
          ! entitiesStorage.save( file ) )
      {
         cerr << "Mesh saving failed." << endl;
         return false;
      }
      return true;
   }

   bool load( tnlFile& file )
   {
      if( ! tnlObject::load( file ) ||
          ! entitiesStorage.load( file ) )
      {
         cerr << "Mesh loading failed." << endl;
         return false;
      }
      return true;
   }

   /*template< int Dimensions >
   struct EntitiesTraits
   {
      typedef tnlDimensionsTag< Dimensions >                       DimensionsTag;
      typedef tnlMeshEntitiesTraits< MeshConfig, DimensionsTag >    MeshEntitiesTraits;
      typedef typename MeshEntitiesTraits::EntityType                       Type;
      typedef typename MeshEntitiesTraits::ContainerType              ContainerType;
      typedef typename MeshEntitiesTraits::SharedContainerType        SharedContainerType;
      typedef typename ContainerType::IndexType                       GlobalIndexType;
      typedef typename ContainerType::ElementType                     EntityType;
      static const bool available = MeshConfig::entityStorage( Dimensions );
   };
   typedef EntitiesTraits< dimensions > CellTraits;*/

   template< int Dimensions >
   bool entitiesAvalable() const
   {
      return MeshTraits::template EntityTraits< Dimensions >::available;
   }

   template< int Dimensions >
   typename MeshTraits::template EntityTraits< Dimensions >::GlobalIndexType getNumberOfEntities() const
   {
      return entitiesStorage.getNumberOfEntities( tnlDimensionsTag< Dimensions >() );
   }

   typename MeshTraits::template EntityTraits< dimensions >::GlobalIndexType getNumberOfCells() const
   {
      return entitiesStorage.getNumberOfEntities( tnlDimensionsTag< dimensions >() );
   }

   template< int Dimensions >
      typename MeshTraits::template EntityTraits< Dimensions >::EntityType&
         getEntity( const typename MeshTraits::template EntityTraits< Dimensions >::GlobalIndexType entityIndex )
   {
      return entitiesStorage.getEntity( tnlDimensionsTag< Dimensions >(), entityIndex );
   }

   template< int Dimensions >
      const typename MeshTraits::template EntityTraits< Dimensions >::EntityType&
         getEntity( const typename MeshTraits::template EntityTraits< Dimensions >::GlobalIndexType entityIndex ) const
   {
      return entitiesStorage.getEntity( tnlDimensionsTag< Dimensions >(), entityIndex );
   }

   template< int Dimensions >
   typename MeshTraits::template EntityTraits< Dimensions >::AccessArrayType& 
   getEntities()
   {
      return entitiesStorage.getEntities( tnlDimensionsTag< Dimensions >() );
   }

   template< int Dimensions >
   const typename MeshTraits::template EntityTraits< Dimensions >::AccessArrayType&
   getEntities() const
   {
      return entitiesStorage.getEntities( tnlDimensionsTag< Dimensions >() );
   }

   typename MeshTraits::template EntityTraits< dimensions >::EntityType&
      getCell( const typename MeshTraits::template EntityTraits< dimensions >::GlobalIndexType entityIndex )
   {
      return entitiesStorage.getEntity( tnlDimensionsTag< dimensions >(), entityIndex );
   }

   const typename MeshTraits::template EntityTraits< dimensions >::EntityType&
      getCell( const typename MeshTraits::template EntityTraits< dimensions >::GlobalIndexType entityIndex ) const
   {
      return entitiesStorage.getEntity( tnlDimensionsTag< dimensions >(), entityIndex );
   }

   void print( ostream& str ) const
   {
      entitiesStorage.print( str );
   }

   bool operator==( const tnlMesh& mesh ) const
   {
      return entitiesStorage.operator==( mesh.entitiesStorage );
   }

   // TODO: this is only for mesh intializer - remove it if possible
   template< typename DimensionsTag >
	typename MeshTraits::template EntityTraits< DimensionsTag::value >::StorageArrayType& entitiesArray()
   {
      return entitiesStorage.entitiesArray( DimensionsTag() ); 
   }
  
   template< typename DimensionsTag, typename SuperDimensionsTag >
	typename tnlMeshTraits< MeshConfig >::GlobalIdArrayType& superentityIdsArray()
   {
      return entitiesStorage.template superentityIdsArray< SuperDimensionsTag >( DimensionsTag() ); 
   }
   
   typedef typename tnlMeshTraits< MeshConfig>::PointArrayType    PointArrayType;
   typedef typename tnlMeshTraits< MeshConfig>::CellSeedArrayType CellSeedArrayType;

   bool init( const PointArrayType& points,
              const CellSeedArrayType& cellSeeds )
   {
      tnlMeshInitializer< MeshConfig> meshInitializer;
      return meshInitializer.createMesh( points, cellSeeds, *this );
   }
   
   protected:
            
      tnlMeshStorageLayers< MeshConfig > entitiesStorage;
      
      tnlMeshConfigValidator< MeshConfig > configValidator;
};


#endif /* TNLMESH_H_ */
