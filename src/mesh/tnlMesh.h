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

template< typename ConfigTag >
class tnlMesh : public tnlObject,
                public tnlMeshStorageLayers< ConfigTag >
{
   //template<typename, typename, typename> friend class InitializerLayer;
   //friend class IOReader<ConfigTag>;

   typedef tnlMeshStorageLayers<ConfigTag>        BaseType;

   public:
   typedef ConfigTag                              Config;
   typedef typename tnlMeshTraits< ConfigTag >::PointType PointType;
   enum { dimensions = tnlMeshTraits< ConfigTag >::meshDimensions };

   static tnlString getType()
   {
      return tnlString( "tnlMesh< ") + ConfigTag::getType() + " >";
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
         return false;
      return true;
   }

   bool load( tnlFile& file )
   {
      if( ! tnlObject::load( file ) ||
          ! BaseType::load( file ) )
         return false;
      return true;
   }

   template< int Dimensions >
   struct EntitiesTraits
   {
      typedef tnlDimensionsTraits< Dimensions >                       DimensionsTraits;
      typedef tnlMeshEntitiesTraits< ConfigTag, DimensionsTraits >    MeshEntitiesTraits;
      typedef typename MeshEntitiesTraits::Type                       Type;
      typedef typename MeshEntitiesTraits::ContainerType              ContainerType;
      typedef typename MeshEntitiesTraits::SharedContainerType        SharedContainerType;
      typedef typename ContainerType::IndexType                       GlobalIndexType;
      typedef typename ContainerType::ElementType                     EntityType;
      enum { available = tnlMeshEntityStorage< ConfigTag, Dimensions >::enabled };
   };

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
      return BaseType::setNumberOfEntities( tnlDimensionsTraits< Dimensions >(), size );
   }

   template< int Dimensions >
   typename EntitiesTraits< Dimensions >::GlobalIndexType getNumberOfEntities() const
   {
      return BaseType::getNumberOfEntities( tnlDimensionsTraits< Dimensions >() );
   }

   bool setNumberOfCells( typename EntitiesTraits< dimensions >::GlobalIndexType size )
   {
      return BaseType::setNumberOfEntities( tnlDimensionsTraits< dimensions >(), size );
   }

   typename EntitiesTraits< dimensions >::GlobalIndexType getNumberOfCells() const
   {
      return BaseType::getNumberOfEntities( tnlDimensionsTraits< dimensions >() );
   }

   template< int Dimensions >
      typename EntitiesTraits< Dimensions >::EntityType&
         getEntity( const typename EntitiesTraits< Dimensions >::GlobalIndexType entityIndex )
   {
      return BaseType::getEntity( tnlDimensionsTraits< Dimensions >(), entityIndex );
   }

   template< int Dimensions >
      const typename EntitiesTraits< Dimensions >::EntityType&
         getEntity( const typename EntitiesTraits< Dimensions >::GlobalIndexType entityIndex ) const
   {
      return BaseType::getEntity( tnlDimensionsTraits< Dimensions >(), entityIndex );
   }

   template< int Dimensions >
      void setEntity( const typename EntitiesTraits< Dimensions >::GlobalIndexType entityIndex,
                      const typename EntitiesTraits< Dimensions >::EntityType& entity )
   {
      BaseType::setEntity( tnlDimensionsTraits< Dimensions >(), entityIndex, entity );
   }

   template< int Dimensions >
   typename EntitiesTraits< Dimensions >::SharedContainerType& getEntities()
   {
      return BaseType::getEntities( tnlDimensionsTraits< Dimensions >() );
   }

   template< int Dimensions >
   const typename EntitiesTraits< Dimensions >::SharedContainerType& getEntities() const
   {
      return BaseType::getEntities( tnlDimensionsTraits< Dimensions >() );
   }

   typename EntitiesTraits< dimensions >::EntityType&
      getCell( const typename EntitiesTraits< dimensions >::GlobalIndexType entityIndex )
   {
      return BaseType::getEntity( tnlDimensionsTraits< dimensions >(), entityIndex );
   }

   const typename EntitiesTraits< dimensions >::EntityType&
      getCell( const typename EntitiesTraits< dimensions >::GlobalIndexType entityIndex ) const
   {
      return BaseType::getEntity( tnlDimensionsTraits< dimensions >(), entityIndex );
   }

   void setCell( const typename EntitiesTraits< dimensions >::GlobalIndexType entityIndex,
                 const typename EntitiesTraits< dimensions >::EntityType& entity )
   {
      BaseType::setEntity( tnlDimensionsTraits< dimensions >(), entityIndex, entity );
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

   tnlStaticAssert( dimensions > 0, "The mesh dimesnions must be greater than 0." );
   tnlStaticAssert( EntitiesTraits< 0 >::available, "Vertices must always be stored" );
   tnlStaticAssert( EntitiesTraits< dimensions >::available, "Cells must always be stored" );
};


#endif /* TNLMESH_H_ */
