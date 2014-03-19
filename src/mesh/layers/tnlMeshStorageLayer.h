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

#include <core/tnlFile.h>
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
   typedef typename Tag::SharedContainerType                            SharedContainerType;
   typedef typename ContainerType::IndexType                            GlobalIndexType;
   typedef typename ContainerType::ElementType                          EntityType;

   protected:

   using BaseType::setNumberOfEntities;
   using BaseType::getNumberOfEntities;
   using BaseType::setEntity;
   using BaseType::getEntity;
   using BaseType::getEntities;

   tnlMeshStorageLayer()
   {
      this->entities.setName( tnlString( "tnlMeshStorageLayer < " ) + tnlString( DimensionsTraits::value ) + " >::entities" );
      this->sharedEntities.setName( tnlString( "tnlMeshStorageLayer < " ) + tnlString( DimensionsTraits::value ) + " >::sharedEntities" );
   }

   /*~tnlMeshStorageLayer()
   {
      cout << "Destroying mesh storage layer with " << DimensionsTraits::value << " dimensions and " << this->entities.getSize() << " entities." << endl;
   }*/

   bool setNumberOfEntities( DimensionsTraits, const GlobalIndexType size )
   {
      if( ! this->entities.setSize( size ) )
         return false;
      this->sharedEntities.bind( this->entities );
      return true;
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

   SharedContainerType& getEntities( DimensionsTraits )
   {
      return this->sharedEntities;
   }

   const SharedContainerType& getEntities( DimensionsTraits ) const
   {
      return this->sharedEntities;
   }

   bool save( tnlFile& file ) const
   {
      if( ! BaseType::save( file ) ||
          ! this->entities.save( file ) )
      {
         cerr << "Saving of the mesh entities with " << DimensionsTraits::value << " dimensions failed." << endl;
         return false;
      }
      return true;
   }

   bool load( tnlFile& file )
   {
      //cout << "Loading mesh layer with dimensions " << DimensionsTraits::value << endl;
      if( ! BaseType::load( file ) ||
          ! this->entities.load( file ) )
      {
         cerr << "Loading of the mesh entities with " << DimensionsTraits::value << " dimensions failed." << endl;
         return false;
      }
      this->sharedEntities.bind( this->entities );
      return true;
   }

   void print( ostream& str ) const
   {
      BaseType::print( str );
      str << "The entities with " << DimensionsTraits::value << " dimensions are: " << endl;
      for( GlobalIndexType i = 0; i < entities.getSize();i ++ )
      {
         str << i << " ";
         entities[ i ].print( str );
         str << endl;
      }
   }

   bool operator==( const tnlMeshStorageLayer& meshLayer ) const
   {
      return ( BaseType::operator==( meshLayer ) && entities == meshLayer.entities );
   }


   protected:
   ContainerType entities;

   SharedContainerType sharedEntities;
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
   typedef typename Tag::SharedContainerType               SharedContainerType;
   typedef typename ContainerType::IndexType               GlobalIndexType;
   typedef typename ContainerType::ElementType             VertexType;
   typedef typename VertexType::PointType                  PointType;

   protected:

   tnlMeshStorageLayer()
   {
      this->vertices.setName( tnlString( "tnlMeshStorageLayer < " ) + tnlString( DimensionsTraits::value ) + " >::vertices" );
      this->sharedVertices.setName( tnlString( "tnlMeshStorageLayer < " ) + tnlString( DimensionsTraits::value ) + " >::sharedVertices" );
   }

   /*~tnlMeshStorageLayer()
   {
        cout << "mesh storage layer: dimensions = " << DimensionsTraits::value << " entities = " << this->vertices.getSize() << endl;
   }*/


   bool setNumberOfVertices( const GlobalIndexType size )
   {
      if( ! this->vertices.setSize( size ) )
         return false;
      this->sharedVertices.bind( this->vertices );
      return true;
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
      return this->vertices[ vertexIndex ];
   }

   const VertexType& getVertex( const GlobalIndexType vertexIndex ) const
   {
      return this->vertices[ vertexIndex ];
   }


   void setVertex( const GlobalIndexType vertexIndex,
                   const PointType& point )
   {
      this->vertices[ vertexIndex ].setPoint( point );
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

   SharedContainerType& getEntities( DimensionsTraits )
   {
      return this->sharedVertices;
   }

   const SharedContainerType& getEntities( DimensionsTraits ) const
   {
      return this->sharedVertices;
   }

   bool save( tnlFile& file ) const
   {
      if( ! this->vertices.save( file ) )
      {
         cerr << "Saving of the mesh entities with " << DimensionsTraits::value << " dimensions failed." << endl;
         return false;
      }
      return true;
   }

   bool load( tnlFile& file )
   {
      if( ! this->vertices.load( file ) )
      {
         cerr << "Loading of the mesh entities with " << DimensionsTraits::value << " dimensions failed." << endl;
         return false;
      }
      this->sharedVertices.bind( this->vertices );
      return true;
   }

   void print( ostream& str ) const
   {
      str << "The mesh vertices are: " << endl;
      for( GlobalIndexType i = 0; i < vertices.getSize();i ++ )
      {
         str << i << vertices[ i ] << endl;
      }
   }

   bool operator==( const tnlMeshStorageLayer& meshLayer ) const
   {
      return ( vertices == meshLayer.vertices );
   }

   private:

   ContainerType vertices;

   SharedContainerType sharedVertices;
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
