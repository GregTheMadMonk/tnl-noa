/***************************************************************************
                          MeshStorageLayer.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/File.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/MeshDetails/traits/MeshEntityTraits.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename DimensionsTag,
          bool EntityStorage = MeshEntityTraits< MeshConfig, DimensionsTag::value >::storageEnabled >
class MeshStorageLayer;


template< typename MeshConfig >
class MeshStorageLayers
   : public MeshStorageLayer< MeshConfig, typename MeshTraits< MeshConfig >::DimensionsTag >
{
};


template< typename MeshConfig,
          typename DimensionsTag >
class MeshStorageLayer< MeshConfig,
                           DimensionsTag,
                           true >
   : public MeshStorageLayer< MeshConfig, typename DimensionsTag::Decrement >,
     public MeshSuperentityStorageLayers< MeshConfig,
                                             typename MeshTraits< MeshConfig >::template EntityTraits< DimensionsTag::value >::EntityTopology >
{
   public:

      static const int Dimensions = DimensionsTag::value;
      typedef MeshStorageLayer< MeshConfig, typename DimensionsTag::Decrement >   BaseType;
      typedef MeshSuperentityStorageLayers< MeshConfig,
                                               typename MeshTraits< MeshConfig >::template EntityTraits< DimensionsTag::value >::EntityTopology > SuperentityStorageBaseType;
      typedef MeshTraits< MeshConfig >                                             MeshTraitsType;
      typedef typename MeshTraitsType::template EntityTraits< Dimensions >         EntityTraitsType;

      typedef typename EntityTraitsType::StorageArrayType                          StorageArrayType;
      typedef typename EntityTraitsType::AccessArrayType                           AccessArrayType;
      typedef typename EntityTraitsType::GlobalIndexType                           GlobalIndexType;
      typedef typename EntityTraitsType::EntityType                                EntityType;
      typedef typename EntityTraitsType::EntityTopology                            EntityTopology;


      using BaseType::getNumberOfEntities;
      using BaseType::getEntity;
      using BaseType::getEntities;

      MeshStorageLayer()
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

      bool save( File& file ) const
      {
         if( ! BaseType::save( file ) ||
             ! this->entities.save( file ) )
         {
            std::cerr << "Saving of the mesh entities with " << DimensionsTag::value << " dimensions failed." << std::endl;
            return false;
         }
         return true;
      }

      bool load( File& file )
      {
         //cout << "Loading mesh layer with dimensions " << DimensionsTag::value << std::endl;
         if( ! BaseType::load( file ) ||
             ! this->entities.load( file ) )
         {
            std::cerr << "Loading of the mesh entities with " << DimensionsTag::value << " dimensions failed." << std::endl;
            return false;
         }
         this->entitiesAccess.bind( this->entities );
         return true;
      }

      void print( std::ostream& str ) const
      {
         BaseType::print( str );
         str << "The entities with " << DimensionsTag::value << " dimensions are: " << std::endl;
         for( GlobalIndexType i = 0; i < entities.getSize();i ++ )
         {
            str << i << " ";
            entities[ i ].print( str );
            str << std::endl;
         }
         SuperentityStorageBaseType::print( str );
      }

      bool operator==( const MeshStorageLayer& meshLayer ) const
      {
         return ( BaseType::operator==( meshLayer ) && entities == meshLayer.entities );
      }


   protected:
      StorageArrayType entities;

      AccessArrayType entitiesAccess;
 
   // TODO: this is only for the mesh initializer - fix it
   public:

      using BaseType::entitiesArray;
 
      typename EntityTraitsType::StorageArrayType& entitiesArray( DimensionsTag )
      {
         return entities;
      }
 
      using BaseType::superentityIdsArray;
	
      template< typename SuperDimensionsTag >
      typename MeshTraitsType::GlobalIdArrayType& superentityIdsArray( DimensionsTag )
      {
         return SuperentityStorageBaseType::superentityIdsArray( SuperDimensionsTag() );
      }
 
      using BaseType::getSuperentityStorageNetwork;
      template< typename SuperdimensionsTag >
      typename MeshTraitsType::template SuperentityTraits< EntityTopology, SuperdimensionsTag::value >::StorageNetworkType&
      getSuperentityStorageNetwork( MeshDimensionsTag< EntityTopology::dimensions > )
      {
         return SuperentityStorageBaseType::getStorageNetwork( SuperdimensionsTag() );
      }
};

template< typename MeshConfig,
          typename DimensionsTag >
class MeshStorageLayer< MeshConfig, DimensionsTag, false >
   : public MeshStorageLayer< MeshConfig, typename DimensionsTag::Decrement  >
{
};

template< typename MeshConfig >
class MeshStorageLayer< MeshConfig, MeshDimensionsTag< 0 >, true > :
   public MeshSuperentityStorageLayers< MeshConfig,
                                           MeshVertexTopology >

{
   public:

   typedef MeshDimensionsTag< 0 >                        DimensionsTag;
 
   typedef MeshSuperentityStorageLayers< MeshConfig,
                                            MeshVertexTopology >     SuperentityStorageBaseType;

   typedef MeshTraits< MeshConfig >                                   MeshTraitsType;
   typedef typename MeshTraitsType::template EntityTraits< 0 >        EntityTraitsType;
 
   typedef typename EntityTraitsType::StorageArrayType                StorageArrayType;
   typedef typename EntityTraitsType::AccessArrayType                 AccessArrayType;
   typedef typename EntityTraitsType::GlobalIndexType                 GlobalIndexType;
   typedef typename EntityTraitsType::EntityType                      VertexType;
   typedef typename VertexType::PointType                             PointType;
   typedef MeshVertexTopology                                         EntityTopology;
 
   MeshStorageLayer()
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

   bool save( File& file ) const
   {
      if( ! this->vertices.save( file ) )
      {
         std::cerr << "Saving of the mesh entities with " << DimensionsTag::value << " dimensions failed." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      if( ! this->vertices.load( file ) )
      {
         std::cerr << "Loading of the mesh entities with " << DimensionsTag::value << " dimensions failed." << std::endl;
         return false;
      }
      this->verticesAccess.bind( this->vertices );
      return true;
   }

   void print( std::ostream& str ) const
   {
      str << "The mesh vertices are: " << std::endl;
      for( GlobalIndexType i = 0; i < vertices.getSize();i ++ )
      {
         str << i << vertices[ i ] << std::endl;
      }
      SuperentityStorageBaseType::print( str );
   }

   bool operator==( const MeshStorageLayer& meshLayer ) const
   {
      return ( vertices == meshLayer.vertices );
   }

   private:

   StorageArrayType vertices;

   AccessArrayType verticesAccess;
 
   // TODO: this is only for the mesh initializer - fix it
   public:
 
      typename EntityTraitsType::StorageArrayType& entitiesArray( DimensionsTag )
      {
         return vertices;
      }

 
      template< typename SuperDimensionsTag >
      typename MeshTraitsType::GlobalIdArrayType& superentityIdsArray( DimensionsTag )
      {
         return SuperentityStorageBaseType::superentityIdsArray( SuperDimensionsTag() );
      }

      template< typename SuperdimensionsTag >
      typename MeshTraitsType::template SuperentityTraits< EntityTopology, SuperdimensionsTag::value >::StorageNetworkType& getSuperentityStorageNetwork( MeshDimensionsTag< EntityTopology::dimensions > )
      {
         return SuperentityStorageBaseType::getStorageNetwork( SuperdimensionsTag() );
      }

};

/****
 * Forces termination of recursive inheritance (prevents compiler from generating huge error logs)
 */
template< typename MeshConfig >
class MeshStorageLayer< MeshConfig, MeshDimensionsTag< 0 >, false >
{
};

} // namespace Meshes
} // namespace TNL
