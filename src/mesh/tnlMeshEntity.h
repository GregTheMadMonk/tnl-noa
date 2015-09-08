/***************************************************************************
                          tnlMeshEntity.h  -  description
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

#ifndef TNLMESHENTITY_H_
#define TNLMESHENTITY_H_

#include <core/tnlFile.h>
#include <core/tnlDynamicTypeTag.h>
#include <mesh/tnlMeshEntityId.h>
#include <mesh/traits/tnlMeshTraits.h>
#include <mesh/tnlDimensionsTag.h>
#include <mesh/topologies/tnlMeshVertexTopology.h>
#include <mesh/layers/tnlMeshSubentityStorageLayer.h>
#include <mesh/layers/tnlMeshSuperentityStorageLayer.h>
#include <mesh/layers/tnlMeshSuperentityAccess.h>
#include <mesh/initializer/tnlMeshEntitySeed.h>

template< typename MeshConfig,
          typename EntityTopology_ >
class tnlMeshEntity
   : public tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology_ >,     
     public tnlMeshSuperentityAccess< MeshConfig, EntityTopology_ >,
     public tnlMeshEntityId< typename MeshConfig::IdType,
                             typename MeshConfig::GlobalIndexType >
{
   public:

      typedef tnlMeshTraits< MeshConfig >                         MeshTraits;
      typedef EntityTopology_                                     EntityTopology;
      typedef typename MeshTraits::GlobalIndexType                GlobalIndexType;
      typedef typename MeshTraits::LocalIndexType                 LocalIndexType;
      typedef typename MeshTraits::IdPermutationArrayAccessorType IdPermutationArrayAccessorType;
      typedef tnlMeshEntitySeed< MeshConfig, EntityTopology >     SeedType;   

      static const int Dimensions = EntityTopology::dimensions;
      static const int MeshDimensions = MeshTraits::meshDimensions;      


      tnlMeshEntity( const SeedType& entitySeed )
      {
         typedef typename SeedType::LocalIndexType LocalIndexType;
         for( LocalIndexType i = 0; i < entitySeed.getCornerIds().getSize(); i++ )
            this->template setSubentityIndex< 0 >( i, entitySeed.getCornerIds()[ i ] );         
      }

      tnlMeshEntity() {}

      static tnlString getType()
      {
         return tnlString( "tnlMesh< " ) +
                           //MeshConfig::getType() + ", " +
                           //EntityTopology::getType() + ", " +
                           " >";
      }

      tnlString getTypeVirtual() const
      {
         return this->getType();
      }

      /*~tnlMeshEntity()
      {
         cerr << "   Destroying entity with " << EntityTopology::dimensions << " dimensions..." << endl;
      }*/

      bool save( tnlFile& file ) const
      {
         if( ! tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology >::save( file ) /*||
             ! tnlMeshSuperentityStorageLayers< MeshConfig, EntityTopology >::save( file )*/ )
            return false;
         return true;
      }

      bool load( tnlFile& file )
      {
         if( ! tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology >::load( file ) /*||
             ! tnlMeshSuperentityStorageLayers< MeshConfig, EntityTopology >::load( file ) */ )
            return false;
         return true;
      }

      void print( ostream& str ) const
      {
         str << "\t Mesh entity dimensions: " << EntityTopology::dimensions << endl;
         tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology >::print( str );
         tnlMeshSuperentityAccess< MeshConfig, EntityTopology >::print( str );
      }

      bool operator==( const tnlMeshEntity& entity ) const
      {
         return ( tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology >::operator==( entity ) &&
                  tnlMeshSuperentityAccess< MeshConfig, EntityTopology >::operator==( entity ) &&
                  tnlMeshEntityId< typename MeshConfig::IdType,
                                   typename MeshConfig::GlobalIndexType >::operator==( entity ) );
      }

      /****
       * Subentities
       */
      template< int SubDimensions > using SubentityTraits = 
      typename MeshTraits::template SubentityTraits< EntityTopology, SubDimensions >;
      
      /*struct SubentityTraits
      {
         static_assert( Dimensions < meshDimensions, "Asking for subentities with more or the same number of dimensions then the mesh itself." );
         typedef tnlDimensionsTag< Dimensions >                 DimensionsTag;
         typedef tnlMeshSubentityTraits< MeshConfig,
                                           EntityTopology,
                                           DimensionsTag::value >      SubentityTraits;
         typedef typename SubentityTraits::ContainerType           ContainerType;
         typedef typename SubentityTraits::SharedContainerType     SharedContainerType;
         typedef typename ContainerType::ElementType               GlobalIndexType;
         typedef int                                               LocalIndexType;

         // TODO: make this as:
         // typedef typename Type::IndexType   LocalIndexType
         //enum { available = tnlMeshSubentityStorage< MeshConfig,
         //                                            EntityTopology,
         //                                            Dimensions >::enabled };
         static const bool available = MeshConfig::template subentityStorage( EntityTopology(), Dimensions );
         enum { subentitiesCount = SubentityTraits::count };
      };*/

      template< int SubDimensions >
      bool subentitiesAvailable() const
      {
         return SubentityTraits< SubDimensions >::available;
      };

      template< int SubDimensions >
      typename SubentityTraits< SubDimensions >::LocalIndexType getNumberOfSubentities() const
      {
         return SubentityTraits< SubDimensions >::count;
      };

      template< int SubDimensions >
      void setSubentityIndex( const LocalIndexType localIndex,
                              const GlobalIndexType globalIndex )
      {
         static_assert( SubentityTraits< SubDimensions >::storageEnabled, "You try to set subentity which is not configured for storage." );
         tnlAssert( 0 <= localIndex &&
                    localIndex < SubentityTraits< SubDimensions >::count,
                    cerr << "localIndex = " << localIndex
                         << " subentitiesCount = "
                         << SubentityTraits< SubDimensions >::count );
         typedef tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology >  SubentityBaseType;
         SubentityBaseType::setSubentityIndex( tnlDimensionsTag< SubDimensions >(),
                                               localIndex,
                                               globalIndex );
      }

      template< int SubDimensions >
      typename SubentityTraits< SubDimensions >::GlobalIndexType
         getSubentityIndex( const typename SubentityTraits< SubDimensions >::LocalIndexType localIndex) const
         {
            static_assert( SubentityTraits< SubDimensions >::storageEnabled, "You try to get subentity which is not configured for storage." );
            tnlAssert( 0 <= localIndex &&
                       localIndex < SubentityTraits< SubDimensions >::count,
                       cerr << "localIndex = " << localIndex
                            << " subentitiesCount = "
                            << SubentityTraits< SubDimensions >::count );
            typedef tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology >  SubentityBaseType;
            return SubentityBaseType::getSubentityIndex( tnlDimensionsTag< SubDimensions >(),
                                                         localIndex );
         }

      template< int SubDimensions >
         typename SubentityTraits< SubDimensions >::SharedContainerType&
            getSubentitiesIndices()
      {
         static_assert( SubentityTraits< SubDimensions >::storageEnabled, "You try to get subentities which are not configured for storage." );
         typedef tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology >  SubentityBaseType;
         return SubentityBaseType::getSubentitiesIndices( tnlDimensionsTag< SubDimensions >() );
      }

      template< int SubDimensions >
         const typename SubentityTraits< SubDimensions >::SharedContainerType&
            getSubentitiesIndices() const
      {
         static_assert( SubentityTraits< SubDimensions >::storageEnabled, "You try to set subentities which are not configured for storage." );
         typedef tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology >  SubentityBaseType;
         return SubentityBaseType::getSubentitiesIndices( tnlDimensionsTag< SubDimensions >() );
      }

      /****
       * Superentities
       */
      template< int SuperDimensions > using SuperentityTraits = 
      typename MeshTraits::template SuperentityTraits< EntityTopology, SuperDimensions >;
      
      /*struct SuperentityTraits
      {
         static_assert( Dimensions <= meshDimensions, "Asking for subentities with more dimensions then the mesh itself." );
         typedef tnlDimensionsTag< Dimensions >                 DimensionsTag;
         typedef tnlMeshSuperentityTraits< MeshConfig,
                                             EntityTopology,
                                             Dimensions >    SuperentityTraits;
         typedef typename SuperentityTraits::ContainerType         ContainerType;
         typedef typename SuperentityTraits::SharedContainerType   SharedContainerType;
         typedef typename ContainerType::ElementType               GlobalIndexType;
         typedef int                                               LocalIndexType;      
         // TODO: make this as:
         // typedef typename Type::IndexType   LocalIndexType      
         static const bool available = MeshConfig::template superentityStorage( EntityTopology(), Dimensions );
      };*/

      /*template< int Dimensions >
      bool setNumberOfSuperentities( const typename SuperentityTraits< Dimensions >::LocalIndexType size )
      {
         static_assert( SuperentityTraits< Dimensions >::available, "You try to set number of superentities which are not configured for storage." );
         tnlAssert( size >= 0,
                    cerr << "size = " << size << endl; );
         typedef tnlMeshSuperentityStorageLayers< MeshConfig, EntityTopology >  SuperentityBaseType;
         return SuperentityBaseType::setNumberOfSuperentities( tnlDimensionsTag< Dimensions >(),
                                                               size );
      }*/

      template< int SuperDimensions >
      typename SuperentityTraits< SuperDimensions >::LocalIndexType getNumberOfSuperentities() const
      {
         static_assert( SuperentityTraits< SuperDimensions >::available, "You try to get number of superentities which are not configured for storage." );
         typedef tnlMeshSuperentityAccess< MeshConfig, EntityTopology >  SuperentityBaseType;
         return SuperentityBaseType::getNumberOfSuperentities( tnlDimensionsTag< SuperDimensions >() );
      }

      template< int SuperDimensions >
      void setSuperentityIndex( const LocalIndexType localIndex,
                                const GlobalIndexType globalIndex )
      {
         static_assert( SuperentityTraits< SuperDimensions >::available, "You try to set superentity which is not configured for storage." );
         tnlAssert( localIndex < this->getNumberOfSuperentities< SuperDimensions >(),
                    cerr << " localIndex = " << localIndex
                         << " this->getNumberOfSuperentities< Dimensions >() = " << this->getNumberOfSuperentities< Dimensions >() << endl; );
         typedef tnlMeshSuperentityAccess< MeshConfig, EntityTopology >  SuperentityBaseType;
         SuperentityBaseType::setSuperentityIndex( tnlDimensionsTag< Dimensions >(),
                                                   localIndex,
                                                   globalIndex );
      }

      template< int SuperDimensions >
      typename SuperentityTraits< SuperDimensions >::GlobalIndexType 
         getSuperentityIndex( const typename SuperentityTraits< SuperDimensions >::LocalIndexType localIndex ) const
      {
         static_assert( SuperentityTraits< SuperDimensions >::available, "You try to get superentity which is not configured for storage." );
         tnlAssert( localIndex < this->getNumberOfSuperentities< SuperDimensions >(),
                    cerr << " localIndex = " << localIndex
                         << " this->getNumberOfSuperentities< Dimensions >() = " << this->getNumberOfSuperentities< SuperDimensions >() << endl; );
         typedef tnlMeshSuperentityAccess< MeshConfig, EntityTopology >  SuperentityBaseType;
         return SuperentityBaseType::getSuperentityIndex( tnlDimensionsTag< SuperDimensions >(),
                                                          localIndex );
      }

      template< int SuperDimensions >
         typename SuperentityTraits< SuperDimensions >::SharedContainerType& getSuperentitiesIndices()
      {
         static_assert( SuperentityTraits< SuperDimensions >::available, "You try to get superentities which are not configured for storage." );
         typedef tnlMeshSuperentityAccess< MeshConfig, EntityTopology >  SuperentityBaseType;
         //return SuperentityBaseType::getSuperentitiesIndices( tnlDimensionsTag< Dimensions >() );
      }

      template< int SuperDimensions >
         const typename SuperentityTraits< SuperDimensions >::SharedContainerType& getSuperentitiesIndices() const
      {
         static_assert( SuperentityTraits< SuperDimensions >::available, "You try to get superentities which are not configured for storage." );
         typedef tnlMeshSuperentityAccess< MeshConfig, EntityTopology >  SuperentityBaseType;
         return SuperentityBaseType::getSubentitiesIndices( tnlDimensionsTag< SuperDimensions >() );
      }

      /****
       * Vertices
       */
      static const int verticesCount = SubentityTraits< 0 >::count;
      typedef typename SubentityTraits< 0 >::ContainerType        ContainerType;
      typedef typename SubentityTraits< 0 >::SharedContainerType  SharedContainerType;
      //typedef typename SubentityTraits< 0 >::GlobalIndexType      GlobalIndexType;
      //typedef typename SubentityTraits< 0 >::LocalIndexType       LocalIndexType;

      LocalIndexType getNumberOfVertices() const
      {
         return verticesCount;
      }

      void setVertexIndex( const LocalIndexType localIndex,
                           const GlobalIndexType globalIndex )
      {
         this->setSubentityIndex< 0 >( localIndex, globalIndex  );
      }

      GlobalIndexType getVertexIndex( const LocalIndexType localIndex ) const
      {
         return this->getSubentityIndex< 0 >( localIndex  );
      }

      SharedContainerType& getVerticesIndices()
      {
         return this->getSubentitiesIndices< 0 >();
      }

      const SharedContainerType& getVerticesIndices() const
      {
         return this->getSubentitiesIndices< 0 >();
      }

      template< int dim >
      IdPermutationArrayAccessorType subentityOrientation( LocalIndexType index ) const
      {
         static const LocalIndexType subentitiesCount = tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, tnlDimensionsTag<dim>>::count;
         tnlAssert( 0 <= index && index < subentitiesCount, );

         return SubentityStorageLayers::subentityOrientation( tnlDimensionsTag< dim >(), index );
      }  

      // TODO: This is only for the mesh initializer, fix this
      typedef tnlMeshSuperentityAccess< MeshConfig, EntityTopology >                     SuperentityAccessBase;
      typedef typename tnlMeshTraits< MeshConfig>::IdArrayAccessorType        IdArrayAccessorType;
      typedef tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology >                SubentityStorageLayers;

      template< typename DimensionsTag >
      typename tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag::value >::IdArrayType& subentityIdsArray()
      {
         return SubentityStorageLayers::subentityIdsArray( DimensionsTag() );
      }

      template<typename DimensionsTag >
      IdArrayAccessorType& superentityIdsArray()
      {
         return SuperentityAccessBase::superentityIdsArray( DimensionsTag());
      }

      template< typename DimensionsTag >
      typename tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag::value >::OrientationArrayType& subentityOrientationsArray()
      {
         return SubentityStorageLayers::subentityOrientationsArray( DimensionsTag() );
      }      
      
};

template< typename MeshConfig >
class tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >
   : public tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology >,
     public tnlMeshEntityId< typename MeshConfig::IdType,
                             typename MeshConfig::GlobalIndexType >
{
   public:

      typedef tnlMeshTraits< MeshConfig >                         MeshTraits;
      typedef tnlMeshVertexTopology                               EntityTopology;
      typedef typename MeshTraits::GlobalIndexType                GlobalIndexType;
      typedef typename MeshTraits::LocalIndexType                 LocalIndexType;
      typedef typename MeshTraits::PointType                      PointType;
      typedef typename MeshTraits::IdPermutationArrayAccessorType IdPermutationArrayAccessorType;
      typedef tnlMeshEntitySeed< MeshConfig, EntityTopology >     SeedType;   

      static const int Dimensions = EntityTopology::dimensions;
      static const int MeshDimensions = MeshTraits::meshDimensions; 
      
      /****
       * The entity typedefs
       */
      /*typedef MeshConfig         MeshMeshConfig;
      typedef tnlMeshVertexTopology  Tag;
      typedef tnlMeshEntitySeed< MeshConfig, tnlMeshVertexTopology >            SeedType;
      typedef typename tnlMeshTraits< MeshConfig >::PointType PointType;
      enum { dimensions = Tag::dimensions };
      enum { meshDimensions = tnlMeshTraits< MeshConfig >::meshDimensions };*/

      /*tnlMeshEntity( const SeedType & entytiSeed )
      {
         typedef typename SeedType::LocalIndexType LocalIndexType;
         for( LocalIndexType i = 0; i < entytiSeed.getCornerIds().getSize(); i++ )
            this->template setSubentityIndex< 0 >( i, entitySeed.getCornerIds()[ i ] );         
      }*/


      
      static tnlString getType()
      {
         return tnlString( "tnlMesh< " ) +
                           //MeshConfig::getType() + ", " +
                           //EntityTopology::getType() + ", " +
                           " >";
      }

      tnlString getTypeVirtual() const
      {
         return this->getType();
      }


      /*~tnlMeshEntity()
      {
         cerr << "   Destroying entity with " << tnlMeshVertexTopology::dimensions << " dimensions..." << endl;
      }*/

      bool save( tnlFile& file ) const
      {
         if( //! tnlMeshSuperentityStorageLayers< MeshConfig, tnlMeshVertexTopology >::save( file ) ||
             ! point.save( file ) )
            return false;
         return true;
      }

      bool load( tnlFile& file )
      {
         if( //! tnlMeshSuperentityStorageLayers< MeshConfig, tnlMeshVertexTopology >::load( file ) ||
             ! point.load( file ) )
            return false;
         return true;
      }

      void print( ostream& str ) const
      {
         str << "\t Mesh entity dimensions: " << tnlMeshVertexTopology::dimensions << endl;
         str << "\t Coordinates = ( " << point << " )";
         tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology >::print( str );
      }

      bool operator==( const tnlMeshEntity& entity ) const
      {
         return ( //tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology >::operator==( entity ) &&
                  tnlMeshEntityId< typename MeshConfig::IdType,
                                   typename MeshConfig::GlobalIndexType >::operator==( entity ) &&
                  point == entity.point );

      }

   /****
    * Superentities
    */
   template< int SuperDimensions > using SuperentityTraits = 
      typename MeshTraits::template SuperentityTraits< EntityTopology, SuperDimensions >;
   /*template< int Dimensions >
   struct SuperentityTraits
   {
      typedef tnlDimensionsTag< Dimensions >                 DimensionsTag;
      typedef tnlMeshSuperentityTraits< MeshConfig,
                                          tnlMeshVertexTopology,
                                          Dimensions >    SuperentityTraits;
      typedef typename SuperentityTraits::StorageArrayType      StorageArrayType;
      typedef typename SuperentityTraits::AccessArrayType       AccessArrayType;
      typedef typename SuperentityTraits::GlobalIndexType       GlobalIndexType;
      typedef int                                               LocalIndexType;
      
      static const bool available = MeshConfig::template superentityStorage< tnlMeshVertexTopology >( Dimensions );
   };*/
   
   /*template< int Dimensions >
   bool setNumberOfSuperentities( const typename SuperentityTraits< Dimensions >::LocalIndexType size )
   {
      tnlAssert( size >= 0,
                 cerr << "size = " << size << endl; );
      typedef tnlMeshSuperentityStorageLayers< MeshConfig, tnlMeshVertexTopology >  SuperentityBaseType;
      return SuperentityBaseType::setNumberOfSuperentities( tnlDimensionsTag< Dimensions >(),
                                                            size );
   }*/

   template< int Dimensions >
   typename SuperentityTraits< Dimensions >::LocalIndexType getNumberOfSuperentities() const
   {
      typedef tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology >  SuperentityBaseType;
      //return SuperentityBaseType::getNumberOfSuperentities( tnlDimensionsTag< Dimensions >() );
   }

   template< int Dimensions >
      typename SuperentityTraits< Dimensions >::SharedContainerType& getSuperentitiesIndices()
   {
      typedef tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology >  SuperentityBaseType;
      //return SuperentityBaseType::getSuperentitiesIndices( tnlDimensionsTag< Dimensions >() );
   }

   template< int Dimensions >
      const typename SuperentityTraits< Dimensions >::SharedContainerType& getSuperentitiesIndeces() const
   {
      typedef tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology >  SuperentityBaseType;
      //return SuperentityBaseType::getSubentitiesIndices( tnlDimensionsTag< Dimensions >() );
   }

   /*template< int Dimensions >
   void setSuperentityIndex( const typename SuperentityTraits< Dimensions >::LocalIndexType localIndex,
                             const typename SuperentityTraits< Dimensions >::GlobalIndexType globalIndex )
   {
      tnlAssert( localIndex < this->getNumberOfSuperentities< Dimensions >(),
                 cerr << " localIndex = " << localIndex
                      << " this->getNumberOfSuperentities< Dimensions >() = " << this->getNumberOfSuperentities< Dimensions >() << endl; );
      typedef tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology >  SuperentityBaseType;
      SuperentityBaseType::setSuperentityIndex( tnlDimensionsTag< Dimensions >(),
                                                localIndex,
                                                globalIndex );
   }*/

   template< int Dimensions >
   typename SuperentityTraits< Dimensions >::GlobalIndexType
      getSuperentityIndex( const typename SuperentityTraits< Dimensions >::LocalIndexType localIndex ) const
   {
      tnlAssert( localIndex < this->getNumberOfSuperentities< Dimensions >(),
                 cerr << " localIndex = " << localIndex
                      << " this->getNumberOfSuperentities< Dimensions >() = " << this->getNumberOfSuperentities< Dimensions >() << endl; );
      typedef tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology >  SuperentityBaseType;
      /*return SuperentityBaseType::getSuperentityIndex( tnlDimensionsTag< Dimensions >(),
                                                       localIndex );*/
   }

   /****
    * Points
    */
   PointType getPoint() const { return this->point; }

   void setPoint( const PointType& point ) { this->point = point; }

   protected:

   PointType point;
   
   
   // TODO: This is only for the mesh initializer, fix this
   public:
   typedef typename tnlMeshTraits< MeshConfig>::IdArrayAccessorType        IdArrayAccessorType;
   typedef tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology > SuperentityAccessBase;
   
   template<typename DimensionsTag >
	IdArrayAccessorType& superentityIdsArray()
	{
		return SuperentityAccessBase::superentityIdsArray( DimensionsTag());
	}

};

template< typename MeshConfig,
          typename EntityTopology >
ostream& operator <<( ostream& str, const tnlMeshEntity< MeshConfig, EntityTopology >& entity )
{
   entity.print( str );
   return str;
}

/****
 * This tells the compiler that theMeshEntity is a type with a dynamic memory allocation.
 * It is necessary for the loading and the saving of the mesh entities arrays.
 */
template< typename MeshConfig,
          typename EntityTopology >
struct tnlDynamicTypeTag< tnlMeshEntity< MeshConfig, EntityTopology > >
{
   enum { value = true };
};

#endif /* TNLMESHENTITY_H_ */
