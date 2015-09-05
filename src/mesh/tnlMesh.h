/***************************************************************************
                          tnlMesh.h  -  description
                             -------------------
    begin                : Feb 16, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
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
#include <mesh/initializer/tnlMeshInitializer.h>

template< typename MeshConfig > //,
          //typename Device = tnlHost >
class tnlMesh : public tnlObject,
                public tnlMeshStorageLayers< MeshConfig >
{
   public:
   typedef MeshConfig                                        Config;
   typedef tnlMeshTraits< MeshConfig >                       MeshTraits;
   typedef typename tnlMeshTraits< MeshConfig >::CellType    CellType;
   typedef typename tnlMeshTraits< MeshConfig >::PointType   PointType;
   static const int dimensions = MeshTraits::meshDimensions;
   template< int Dimensions > using EntityTraits = typename MeshTraits::template EntityTraits< Dimensions >;

   static tnlString getType();
   
   virtual tnlString getTypeVirtual() const;
   
   static constexpr int getDimensions();

   template< int Dimensions >
   bool entitiesAvalable() const;

   // TODO: jeden GlobalIndexType a LocalIndexType pro vsechny entity
   
   typename EntityTraits< dimensions >::GlobalIndexType getNumberOfCells() const;

   template< int Dimensions >
   typename EntityTraits< Dimensions >::GlobalIndexType getNumberOfEntities() const;

   CellType& getCell( const typename MeshTraits::template EntityTraits< dimensions >::GlobalIndexType entityIndex );

   const CellType& getCell( const typename MeshTraits::template EntityTraits< dimensions >::GlobalIndexType entityIndex ) const;

   

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );   
   



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
   
   
   using tnlObject::save;
   using tnlObject::load;
   
   protected:
            
      tnlMeshStorageLayers< MeshConfig > entitiesStorage;
      
      tnlMeshConfigValidator< MeshConfig > configValidator;
};


#include <mesh/tnlMesh_impl.h>

#endif /* TNLMESH_H_ */
