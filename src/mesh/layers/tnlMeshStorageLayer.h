/***************************************************************************
                          tnlMeshStorageLayer.h  -  description
                             -------------------
    begin                : Feb 11, 2014
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

#ifndef TNLMESHSTORAGELAYER_H_
#define TNLMESHSTORAGELAYER_H_

#include <mesh/traits/tnlMeshTraits.h>
#include <mesh/traits/tnlMeshEntitiesTraits.h>
#include <mesh/traits/tnlStorageTraits.h>

template< typename DimensionsTraits,
          typename Device,
          typename ConfigTag,
          typename EntityStorageTag = typename tnlMeshEntitiesTraits< ConfigTag,
                                                                      DimensionsTraits >::EntityStorageTag >
class tnlMeshStorageTag;

template< typename ConfigTag,
          typename DimensionsTraits,
          typename EntityStorageTag = typename tnlMeshEntitiesTraits< ConfigTag,
                                                                      DimensionsTraits >::EntityStorageTag >
class tnlMeshStorageLayer;


template< typename ConfigTag >
class tnlMeshStorageLayers
   : public tnlMeshStorageLayer< ConfigTag,
                                 typename tnlMeshTraits< ConfigTag >::DimensionsTraits >
{};


template< typename ConfigTag,
          typename DimensionsTraits >
class tnlMeshStorageLayer< ConfigTag,
                           DimensionsTraits,
                           tnlStorageTraits< true > >
   : public tnlMeshStorageLayer< ConfigTag, typename DimensionsTraits::Previous >
{
   typedef tnlMeshStorageLayer< ConfigTag,
                                typename DimensionsTraits::Previous >   BaseType;

   typedef tnlMeshEntitiesTraits< ConfigTag, DimensionsTraits >         Tag;
   typedef typename Tag::ContainerType                                  ContainerType;
   //typedef typename Tag::SharedArrayType                              SharedArrayType;
   typedef typename ContainerType::IndexType                            GlobalIndexType;
   typedef typename ContainerType::ElementType                          EntityType;

   protected:

   using BaseType::setNumberOfEntities;
   using BaseType::getNumberOfEntities;
   using BaseType::setEntity;
   using BaseType::getEntity;

   bool setNumberOfEntities( DimensionsTraits, const GlobalIndexType size )
   {
      return this->entities.setSize( size );
   }

   GlobalIndexType getNumberOfEntities( DimensionsTraits ) const
   {
      return this->entities.getSize();
   }

   void setEntity( DimensionsTraits,
                   const GlobalIndexType entityIndex,
                   const EntityType& entity ) const
   {
      this->entities.setElement( entityIndex, entity );
   }

   EntityType& getEntity( DimensionsTraits,
                          const GlobalIndexType entityIndex )
   {
      return this->entities[ entityIndex ];
   }

   const EntityType& getEntity( DimensionsTraits,
                                const GlobalIndexType entityIndex ) const
   {
      return this->entities[ entityIndex ];
   }

   private:
   ContainerType entities;
};

template< typename ConfigTag,
          typename DimensionsTraits >
class tnlMeshStorageLayer< ConfigTag,
                           DimensionsTraits,
                           tnlStorageTraits< false > >
   : public tnlMeshStorageLayer< ConfigTag,
                                 typename DimensionsTraits::Previous >
{
};

template< typename ConfigTag >
class tnlMeshStorageLayer< ConfigTag,
                           tnlDimensionsTraits< 0 >,
                           tnlStorageTraits< true > >
{
   typedef tnlDimensionsTraits< 0 >                        DimensionsTraits;

   typedef tnlMeshEntitiesTraits< ConfigTag,
                                  DimensionsTraits >       Tag;
   typedef typename Tag::ContainerType                     ContainerType;
   typedef typename ContainerType::IndexType               GlobalIndexType;
   typedef typename ContainerType::ElementType             VertexType;
   typedef typename VertexType::PointType                  PointType;
   //typedef typename Tag::SharedArrayType                 SharedArrayType;

   protected:

   bool setNumberOfVertices( const GlobalIndexType size )
   {
      return this->vertices.setSize( size );
   }

   GlobalIndexType getNumberOfVertices() const
   {
      return this->vertices.getSize();
   }

   void setVertex( const GlobalIndexType vertexIndex,
                   const VertexType& vertex ) const
   {
      this->vertices.setElement( vertexIndex, vertex );
   }

   VertexType& getVertex( const GlobalIndexType vertexIndex )
   {
      return this->vertices.getElement( vertexIndex );
   }

   const VertexType& getVertex( const GlobalIndexType vertexIndex ) const
   {
      return this->vertices.getElement( vertexIndex );
   }


   void setVertex( const GlobalIndexType vertexIndex,
                   const PointType& point ) const
   {
      this->vertices.getElement( vertexIndex ).setPoint( point );
   }

   /****
    * This is only for the completeness and compatibility
    * with higher dimensions entities storage layers.
    */
   bool setNumberOfEntities( DimensionsTraits,
                             const GlobalIndexType size )
   {
      return this->vertices.setSize( size );
   }

   GlobalIndexType getNumberOfEntities( DimensionsTraits ) const
   {
      return this->vertices.getSize();
   }

   void setEntity( DimensionsTraits,
                   const GlobalIndexType entityIndex,
                   const VertexType& entity ) const
   {
      this->vertices.setElement( entityIndex, entity );
   }

   const VertexType& getEntity( DimensionsTraits,
                                const GlobalIndexType entityIndex ) const
   {
      return this->vertices.getElement( entityIndex );
   }

   private:

   ContainerType vertices;
};

/****
 * Forces termination of recursive inheritance (prevents compiler from generating huge error logs)
 */
template< typename ConfigTag >
class tnlMeshStorageLayer< ConfigTag,
                           tnlDimensionsTraits< 0 >,
                           tnlStorageTraits< false > >
{
   protected:

   void setNumberOfEntities();
};


#endif /* TNLMESHSTORAGELAYER_H_ */
