/***************************************************************************
                          tnlMeshStorageLayer.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <core/tnlFile.h>
#include <mesh/traits/tnlMeshTraits.h>
#include <mesh/traits/tnlMeshEntityTraits.h>
#include <mesh/traits/tnlMeshTraits.h>

namespace TNL {

template< typename MeshConfig,
          typename DimensionsTag,
          bool EntityStorage = tnlMeshEntityTraits< MeshConfig, DimensionsTag::value >::storageEnabled >
class tnlMeshStorageLayer;


template< typename MeshConfig >
class tnlMeshStorageLayers
   : public tnlMeshStorageLayer< MeshConfig, typename tnlMeshTraits< MeshConfig >::DimensionsTag >
{
};


template< typename MeshConfig,
          typename DimensionsTag >
class tnlMeshStorageLayer< MeshConfig,
                           DimensionsTag,
                           true >
   : public tnlMeshStorageLayer< MeshConfig, typename DimensionsTag::Decrement >,
     public tnlMeshSuperentityStorageLayers< MeshConfig,
                                             typename tnlMeshTraits< MeshConfig >::template EntityTraits< DimensionsTag::value >::EntityTopology >
{
   public:

      static const int Dimensions = DimensionsTag::value;
      typedef tnlMeshStorageLayer< MeshConfig, typename DimensionsTag::Decrement >   BaseType;
      typedef tnlMeshSuperentityStorageLayers< MeshConfig,
                                               typename tnlMeshTraits< MeshConfig >::template EntityTraits< DimensionsTag::value >::EntityTopology > SuperentityStorageBaseType;
      typedef tnlMeshTraits< MeshConfig >                                          MeshTraits;
      typedef typename MeshTraits::template EntityTraits< Dimensions >             EntityTraits;

      typedef typename EntityTraits::StorageArrayType                              StorageArrayType;
      typedef typename EntityTraits::AccessArrayType                               AccessArrayType;
      typedef typename EntityTraits::GlobalIndexType                               GlobalIndexType;
      typedef typename EntityTraits::EntityType                                    EntityType;
      typedef typename EntityTraits::EntityTopology                                EntityTopology;


      using BaseType::getNumberOfEntities;
      using BaseType::getEntity;
      using BaseType::getEntities;

      tnlMeshStorageLayer()
      {
      }

      GlobalIndexType getNumberOfEntities( DimensionsTag ) const
      {
         return this->entities.getSize();
      }

      EntityType& getEntity( DimensionsTag,
                             const GlobalIndexType entityIndex )
      {
         return this->entities[ entityIndex ];
      }

      const EntityType& getEntity( DimensionsTag,
                                   const GlobalIndexType entityIndex ) const
      {
         return this->entities[ entityIndex ];
      }

      AccessArrayType& getEntities( DimensionsTag )
      {
         return this->sharedEntities;
      }

      const AccessArrayType& getEntities( DimensionsTag ) const
      {
         return this->sharedEntities;
      }

      bool save( tnlFile& file ) const
      {
         if( ! BaseType::save( file ) ||
             ! this->entities.save( file ) )
         {
            cerr << "Saving of the mesh entities with " << DimensionsTag::value << " dimensions failed." << endl;
            return false;
         }
         return true;
      }

      bool load( tnlFile& file )
      {
         //cout << "Loading mesh layer with dimensions " << DimensionsTag::value << endl;
         if( ! BaseType::load( file ) ||
             ! this->entities.load( file ) )
         {
            cerr << "Loading of the mesh entities with " << DimensionsTag::value << " dimensions failed." << endl;
            return false;
         }
         this->entitiesAccess.bind( this->entities );
         return true;
      }

      void print( ostream& str ) const
      {
         BaseType::print( str );
         str << "The entities with " << DimensionsTag::value << " dimensions are: " << endl;
         for( GlobalIndexType i = 0; i < entities.getSize();i ++ )
         {
            str << i << " ";
            entities[ i ].print( str );
            str << endl;
         }
         SuperentityStorageBaseType::print( str );
      }

      bool operator==( const tnlMeshStorageLayer& meshLayer ) const
      {
         return ( BaseType::operator==( meshLayer ) && entities == meshLayer.entities );
      }


   protected:
      StorageArrayType entities;

      AccessArrayType entitiesAccess;
 
   // TODO: this is only for the mesh initializer - fix it
   public:

      using BaseType::entitiesArray;
 
      typename EntityTraits::StorageArrayType& entitiesArray( DimensionsTag )
      {
         return entities;
      }
 
      using BaseType::superentityIdsArray;
	
      template< typename SuperDimensionsTag >
      typename MeshTraits::GlobalIdArrayType& superentityIdsArray( DimensionsTag )
      {
         return SuperentityStorageBaseType::superentityIdsArray( SuperDimensionsTag() );
      }
 
      using BaseType::getSuperentityStorageNetwork;
      template< typename SuperdimensionsTag >
      typename MeshTraits::template SuperentityTraits< EntityTopology, SuperdimensionsTag::value >::StorageNetworkType&
      getSuperentityStorageNetwork( tnlDimensionsTag< EntityTopology::dimensions > )
      {
         return SuperentityStorageBaseType::getStorageNetwork( SuperdimensionsTag() );
      }
};

template< typename MeshConfig,
          typename DimensionsTag >
class tnlMeshStorageLayer< MeshConfig, DimensionsTag, false >
   : public tnlMeshStorageLayer< MeshConfig, typename DimensionsTag::Decrement  >
{
};

template< typename MeshConfig >
class tnlMeshStorageLayer< MeshConfig, tnlDimensionsTag< 0 >, true > :
   public tnlMeshSuperentityStorageLayers< MeshConfig,
                                           tnlMeshVertexTopology >

{
   public:

   typedef tnlDimensionsTag< 0 >                        DimensionsTag;
 
   typedef tnlMeshSuperentityStorageLayers< MeshConfig,
                                            tnlMeshVertexTopology > SuperentityStorageBaseType;

   typedef tnlMeshTraits< MeshConfig >                                          MeshTraits;
   typedef typename MeshTraits::template EntityTraits< 0 >                      EntityTraits;
 
   typedef typename EntityTraits::StorageArrayType                              StorageArrayType;
   typedef typename EntityTraits::AccessArrayType                               AccessArrayType;
   typedef typename EntityTraits::GlobalIndexType                               GlobalIndexType;
   typedef typename EntityTraits::EntityType                                    VertexType;
   typedef typename VertexType::PointType                                       PointType;
   typedef tnlMeshVertexTopology                                                EntityTopology;
 
   tnlMeshStorageLayer()
   {
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

   GlobalIndexType getNumberOfEntities( DimensionsTag ) const
   {
      return this->vertices.getSize();
   }

   VertexType& getEntity( DimensionsTag,
                          const GlobalIndexType entityIndex )
   {
      return this->vertices[ entityIndex ];
   }

   const VertexType& getEntity( DimensionsTag,
                                const GlobalIndexType entityIndex ) const
   {
      return this->vertices.getElement( entityIndex );
   }

   AccessArrayType& getEntities( DimensionsTag )
   {
      return this->sharedVertices;
   }

   const AccessArrayType& getEntities( DimensionsTag ) const
   {
      return this->sharedVertices;
   }

   bool save( tnlFile& file ) const
   {
      if( ! this->vertices.save( file ) )
      {
         cerr << "Saving of the mesh entities with " << DimensionsTag::value << " dimensions failed." << endl;
         return false;
      }
      return true;
   }

   bool load( tnlFile& file )
   {
      if( ! this->vertices.load( file ) )
      {
         cerr << "Loading of the mesh entities with " << DimensionsTag::value << " dimensions failed." << endl;
         return false;
      }
      this->verticesAccess.bind( this->vertices );
      return true;
   }

   void print( ostream& str ) const
   {
      str << "The mesh vertices are: " << endl;
      for( GlobalIndexType i = 0; i < vertices.getSize();i ++ )
      {
         str << i << vertices[ i ] << endl;
      }
      SuperentityStorageBaseType::print( str );
   }

   bool operator==( const tnlMeshStorageLayer& meshLayer ) const
   {
      return ( vertices == meshLayer.vertices );
   }

   private:

   StorageArrayType vertices;

   AccessArrayType verticesAccess;
 
   // TODO: this is only for the mesh initializer - fix it
   public:
 
      typename EntityTraits::StorageArrayType& entitiesArray( DimensionsTag )
      {
         return vertices;
      }

 
      template< typename SuperDimensionsTag >
      typename MeshTraits::GlobalIdArrayType& superentityIdsArray( DimensionsTag )
      {
         return SuperentityStorageBaseType::superentityIdsArray( SuperDimensionsTag() );
      }

      template< typename SuperdimensionsTag >
      typename MeshTraits::template SuperentityTraits< EntityTopology, SuperdimensionsTag::value >::StorageNetworkType& getSuperentityStorageNetwork( tnlDimensionsTag< EntityTopology::dimensions > )
      {
         return SuperentityStorageBaseType::getStorageNetwork( SuperdimensionsTag() );
      }

};

/****
 * Forces termination of recursive inheritance (prevents compiler from generating huge error logs)
 */
template< typename MeshConfig >
class tnlMeshStorageLayer< MeshConfig, tnlDimensionsTag< 0 >, false >
{
};

} // namespace TNL