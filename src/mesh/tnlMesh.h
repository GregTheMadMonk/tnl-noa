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
#include <mesh/layers/tnlMeshStorageLayer.h>
#include <mesh/config/tnlMeshConfigValidator.h>

template< typename MeshConfig >
class tnlMesh : public tnlObject,
                public tnlMeshStorageLayers< MeshConfig >
{
   typedef tnlMeshStorageLayers< MeshConfig >                BaseType;

   public:
   typedef MeshConfig                                        Config;
   typedef typename tnlMeshTraits< MeshConfig >::PointType   PointType;
   enum { dimensions = tnlMeshTraits< MeshConfig >::meshDimensions };

   /*~tnlMesh()
   {
      cerr << "Destroying mesh " << this->getName() << endl;
   }*/

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
          ! BaseType::save( file ) )
      {
         cerr << "Mesh saving failed." << endl;
         return false;
      }
      return true;
   }

   bool load( tnlFile& file )
   {
      if( ! tnlObject::load( file ) ||
          ! BaseType::load( file ) )
      {
         cerr << "Mesh loading failed." << endl;
         return false;
      }
      return true;
   }

   template< int Dimensions >
   struct EntitiesTraits
   {
      typedef tnlDimensionsTag< Dimensions >                       DimensionsTag;
      typedef tnlMeshEntitiesTraits< MeshConfig, DimensionsTag >    MeshEntitiesTraits;
      typedef typename MeshEntitiesTraits::Type                       Type;
      typedef typename MeshEntitiesTraits::ContainerType              ContainerType;
      typedef typename MeshEntitiesTraits::SharedContainerType        SharedContainerType;
      typedef typename ContainerType::IndexType                       GlobalIndexType;
      typedef typename ContainerType::ElementType                     EntityType;
      static const bool available = MeshConfig::entityStorage( Dimensions );
   };
   typedef EntitiesTraits< dimensions > CellTraits;

   using BaseType::setNumberOfVertices;
   using BaseType::getNumberOfVertices;
   using BaseType::setVertex;
   using BaseType::getVertex;

   template< int Dimensions >
   bool entitiesAvalable() const
   {
      return EntitiesTraits< Dimensions >::available;
   }

   template< int Dimensions >
   bool setNumberOfEntities( typename EntitiesTraits< Dimensions >::GlobalIndexType size )
   {
      return BaseType::setNumberOfEntities( tnlDimensionsTag< Dimensions >(), size );
   }

   template< int Dimensions >
   typename EntitiesTraits< Dimensions >::GlobalIndexType getNumberOfEntities() const
   {
      return BaseType::getNumberOfEntities( tnlDimensionsTag< Dimensions >() );
   }

   bool setNumberOfCells( typename EntitiesTraits< dimensions >::GlobalIndexType size )
   {
      return BaseType::setNumberOfEntities( tnlDimensionsTag< dimensions >(), size );
   }

   typename EntitiesTraits< dimensions >::GlobalIndexType getNumberOfCells() const
   {
      return BaseType::getNumberOfEntities( tnlDimensionsTag< dimensions >() );
   }

   template< int Dimensions >
      typename EntitiesTraits< Dimensions >::EntityType&
         getEntity( const typename EntitiesTraits< Dimensions >::GlobalIndexType entityIndex )
   {
      return BaseType::getEntity( tnlDimensionsTag< Dimensions >(), entityIndex );
   }

   template< int Dimensions >
      const typename EntitiesTraits< Dimensions >::EntityType&
         getEntity( const typename EntitiesTraits< Dimensions >::GlobalIndexType entityIndex ) const
   {
      return BaseType::getEntity( tnlDimensionsTag< Dimensions >(), entityIndex );
   }

   template< int Dimensions >
      void setEntity( const typename EntitiesTraits< Dimensions >::GlobalIndexType entityIndex,
                      const typename EntitiesTraits< Dimensions >::EntityType& entity )
   {
      BaseType::setEntity( tnlDimensionsTag< Dimensions >(), entityIndex, entity );
   }

   template< int Dimensions >
   typename EntitiesTraits< Dimensions >::SharedContainerType& getEntities()
   {
      return BaseType::getEntities( tnlDimensionsTag< Dimensions >() );
   }

   template< int Dimensions >
   const typename EntitiesTraits< Dimensions >::SharedContainerType& getEntities() const
   {
      return BaseType::getEntities( tnlDimensionsTag< Dimensions >() );
   }

   typename EntitiesTraits< dimensions >::EntityType&
      getCell( const typename EntitiesTraits< dimensions >::GlobalIndexType entityIndex )
   {
      return BaseType::getEntity( tnlDimensionsTag< dimensions >(), entityIndex );
   }

   const typename EntitiesTraits< dimensions >::EntityType&
      getCell( const typename EntitiesTraits< dimensions >::GlobalIndexType entityIndex ) const
   {
      return BaseType::getEntity( tnlDimensionsTag< dimensions >(), entityIndex );
   }

   void setCell( const typename EntitiesTraits< dimensions >::GlobalIndexType entityIndex,
                 const typename EntitiesTraits< dimensions >::EntityType& entity )
   {
      BaseType::setEntity( tnlDimensionsTag< dimensions >(), entityIndex, entity );
   }

   void print( ostream& str ) const
   {
      BaseType::print( str );
   }

   bool operator==( const tnlMesh& mesh ) const
   {
      return BaseType::operator==( mesh );
   }

   private:

   void init();

   tnlMeshConfigValidator< MeshConfig > configValidator;
};


#endif /* TNLMESH_H_ */
