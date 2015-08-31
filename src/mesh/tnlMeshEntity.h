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
#include <mesh/tnlMeshEntitySeed.h>

template< typename MeshConfig,
          typename EntityTag >
class tnlMeshEntity
   : public tnlMeshSubentityStorageLayers< MeshConfig, EntityTag >,     
     public tnlMeshSuperentityAccess< MeshConfig, EntityTag >,
     public tnlMeshEntityId< typename MeshConfig::IdType,
                             typename MeshConfig::GlobalIndexType >
{
   public:

   /****
    * Entity typedefs
    */
   typedef MeshConfig                                            MeshMeshConfig;
   typedef EntityTag                                            Tag;
   typedef tnlMeshEntitySeed< MeshConfig, EntityTag >            SeedType;
   enum { dimensions = Tag::dimensions };
   enum { meshDimensions = tnlMeshTraits< MeshConfig >::meshDimensions };      
   typedef typename tnlMeshTraits< MeshConfig>::IdPermutationArrayAccessorType IdPermutationArrayAccessorType;

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
                        //EntityTag::getType() + ", " +
                        " >";
   }

   tnlString getTypeVirtual() const
   {
      return this->getType();
   }

   /*~tnlMeshEntity()
   {
      cerr << "   Destroying entity with " << EntityTag::dimensions << " dimensions..." << endl;
   }*/

   bool save( tnlFile& file ) const
   {
      if( ! tnlMeshSubentityStorageLayers< MeshConfig, EntityTag >::save( file ) /*||
          ! tnlMeshSuperentityStorageLayers< MeshConfig, EntityTag >::save( file )*/ )
         return false;
      return true;
   }

   bool load( tnlFile& file )
   {
      if( ! tnlMeshSubentityStorageLayers< MeshConfig, EntityTag >::load( file ) /*||
          ! tnlMeshSuperentityStorageLayers< MeshConfig, EntityTag >::load( file ) */ )
         return false;
      return true;
   }

   void print( ostream& str ) const
   {
      str << "\t Mesh entity dimensions: " << EntityTag::dimensions << endl;
      tnlMeshSubentityStorageLayers< MeshConfig, EntityTag >::print( str );
      tnlMeshSuperentityAccess< MeshConfig, EntityTag >::print( str );
   }

   bool operator==( const tnlMeshEntity& entity ) const
   {
      return ( tnlMeshSubentityStorageLayers< MeshConfig, EntityTag >::operator==( entity ) &&
               tnlMeshSuperentityAccess< MeshConfig, EntityTag >::operator==( entity ) &&
               tnlMeshEntityId< typename MeshConfig::IdType,
                                typename MeshConfig::GlobalIndexType >::operator==( entity ) );
   }

   /****
    * Subentities
    */
   template< int Dimensions >
   struct SubentitiesTraits
   {
      static_assert( Dimensions < meshDimensions, "Asking for subentities with more or the same number of dimensions then the mesh itself." );
      typedef tnlDimensionsTag< Dimensions >                 DimensionsTag;
      typedef tnlMeshSubentityTraits< MeshConfig,
                                        EntityTag,
                                        DimensionsTag::value >      SubentityTraits;
      typedef typename SubentityTraits::ContainerType           ContainerType;
      typedef typename SubentityTraits::SharedContainerType     SharedContainerType;
      typedef typename ContainerType::ElementType               GlobalIndexType;
      typedef int                                               LocalIndexType;

      // TODO: make this as:
      // typedef typename Type::IndexType   LocalIndexType
      /*enum { available = tnlMeshSubentityStorage< MeshConfig,
                                                  EntityTag,
                                                  Dimensions >::enabled };*/
      static const bool available = MeshConfig::template subentityStorage( EntityTag(), Dimensions );
      enum { subentitiesCount = SubentityTraits::count };
   };

   template< int Dimensions >
   bool subentitiesAvailable() const
   {
      return SubentitiesTraits< Dimensions >::available;
   };

   template< int Dimensions >
   typename SubentitiesTraits< Dimensions >::LocalIndexType getNumberOfSubentities() const
   {
      return SubentitiesTraits< Dimensions >::subentitiesCount;
   };

   template< int Dimensions >
   void setSubentityIndex( const typename SubentitiesTraits< Dimensions >::LocalIndexType localIndex,
                           const typename SubentitiesTraits< Dimensions >::GlobalIndexType globalIndex )
   {
      static_assert( SubentitiesTraits< Dimensions >::available, "You try to set subentity which is not configured for storage." );
      tnlAssert( 0 <= localIndex &&
                 localIndex < SubentitiesTraits< Dimensions >::subentitiesCount,
                 cerr << "localIndex = " << localIndex
                      << " subentitiesCount = "
                      << SubentitiesTraits< Dimensions >::subentitiesCount );
      typedef tnlMeshSubentityStorageLayers< MeshConfig, EntityTag >  SubentityBaseType;
      SubentityBaseType::setSubentityIndex( tnlDimensionsTag< Dimensions >(),
                                            localIndex,
                                            globalIndex );
   }

   template< int Dimensions >
   typename SubentitiesTraits< Dimensions >::GlobalIndexType
      getSubentityIndex( const typename SubentitiesTraits< Dimensions >::LocalIndexType localIndex) const
      {
         static_assert( SubentitiesTraits< Dimensions >::available, "You try to get subentity which is not configured for storage." );
         tnlAssert( 0 <= localIndex &&
                    localIndex < SubentitiesTraits< Dimensions >::subentitiesCount,
                    cerr << "localIndex = " << localIndex
                         << " subentitiesCount = "
                         << SubentitiesTraits< Dimensions >::subentitiesCount );
         typedef tnlMeshSubentityStorageLayers< MeshConfig, EntityTag >  SubentityBaseType;
         return SubentityBaseType::getSubentityIndex( tnlDimensionsTag< Dimensions >(),
                                                      localIndex );
      }

   template< int Dimensions >
      typename SubentitiesTraits< Dimensions >::SharedContainerType&
         getSubentitiesIndices()
   {
      static_assert( SubentitiesTraits< Dimensions >::available, "You try to get subentities which are not configured for storage." );
      typedef tnlMeshSubentityStorageLayers< MeshConfig, EntityTag >  SubentityBaseType;
      return SubentityBaseType::getSubentitiesIndices( tnlDimensionsTag< Dimensions >() );
   }

   template< int Dimensions >
      const typename SubentitiesTraits< Dimensions >::SharedContainerType&
         getSubentitiesIndices() const
   {
      static_assert( SubentitiesTraits< Dimensions >::available, "You try to set subentities which are not configured for storage." );
      typedef tnlMeshSubentityStorageLayers< MeshConfig, EntityTag >  SubentityBaseType;
      return SubentityBaseType::getSubentitiesIndices( tnlDimensionsTag< Dimensions >() );
   }

   /****
    * Superentities
    */
   template< int Dimensions >
   struct SuperentitiesTraits
   {
      static_assert( Dimensions <= meshDimensions, "Asking for subentities with more dimensions then the mesh itself." );
      typedef tnlDimensionsTag< Dimensions >                 DimensionsTag;
      typedef tnlMeshSuperentityTraits< MeshConfig,
                                          EntityTag,
                                          Dimensions >    SuperentityTraits;
      typedef typename SuperentityTraits::ContainerType         ContainerType;
      typedef typename SuperentityTraits::SharedContainerType   SharedContainerType;
      typedef typename ContainerType::ElementType               GlobalIndexType;
      typedef int                                               LocalIndexType;      
      // TODO: make this as:
      // typedef typename Type::IndexType   LocalIndexType      
      static const bool available = MeshConfig::template superentityStorage( EntityTag(), Dimensions );
   };

   /*template< int Dimensions >
   bool setNumberOfSuperentities( const typename SuperentitiesTraits< Dimensions >::LocalIndexType size )
   {
      static_assert( SuperentitiesTraits< Dimensions >::available, "You try to set number of superentities which are not configured for storage." );
      tnlAssert( size >= 0,
                 cerr << "size = " << size << endl; );
      typedef tnlMeshSuperentityStorageLayers< MeshConfig, EntityTag >  SuperentityBaseType;
      return SuperentityBaseType::setNumberOfSuperentities( tnlDimensionsTag< Dimensions >(),
                                                            size );
   }*/

   template< int Dimensions >
   typename SuperentitiesTraits< Dimensions >::LocalIndexType getNumberOfSuperentities() const
   {
      static_assert( SuperentitiesTraits< Dimensions >::available, "You try to get number of superentities which are not configured for storage." );
      typedef tnlMeshSuperentityAccess< MeshConfig, EntityTag >  SuperentityBaseType;
      return SuperentityBaseType::getNumberOfSuperentities( tnlDimensionsTag< Dimensions >() );
   }

   template< int Dimensions >
   void setSuperentityIndex( const typename SuperentitiesTraits< Dimensions >::LocalIndexType localIndex,
                             const typename SuperentitiesTraits< Dimensions >::GlobalIndexType globalIndex )
   {
      static_assert( SuperentitiesTraits< Dimensions >::available, "You try to set superentity which is not configured for storage." );
      tnlAssert( localIndex < this->getNumberOfSuperentities< Dimensions >(),
                 cerr << " localIndex = " << localIndex
                      << " this->getNumberOfSuperentities< Dimensions >() = " << this->getNumberOfSuperentities< Dimensions >() << endl; );
      typedef tnlMeshSuperentityAccess< MeshConfig, EntityTag >  SuperentityBaseType;
      SuperentityBaseType::setSuperentityIndex( tnlDimensionsTag< Dimensions >(),
                                                localIndex,
                                                globalIndex );
   }

   template< int Dimensions >
   typename SuperentitiesTraits< Dimensions >::GlobalIndexType 
      getSuperentityIndex( const typename SuperentitiesTraits< Dimensions >::LocalIndexType localIndex ) const
   {
      static_assert( SuperentitiesTraits< Dimensions >::available, "You try to get superentity which is not configured for storage." );
      tnlAssert( localIndex < this->getNumberOfSuperentities< Dimensions >(),
                 cerr << " localIndex = " << localIndex
                      << " this->getNumberOfSuperentities< Dimensions >() = " << this->getNumberOfSuperentities< Dimensions >() << endl; );
      typedef tnlMeshSuperentityAccess< MeshConfig, EntityTag >  SuperentityBaseType;
      return SuperentityBaseType::getSuperentityIndex( tnlDimensionsTag< Dimensions >(),
                                                       localIndex );
   }

   template< int Dimensions >
      typename SuperentitiesTraits< Dimensions >::SharedContainerType& getSuperentitiesIndices()
   {
      static_assert( SuperentitiesTraits< Dimensions >::available, "You try to get superentities which are not configured for storage." );
      typedef tnlMeshSuperentityAccess< MeshConfig, EntityTag >  SuperentityBaseType;
      //return SuperentityBaseType::getSuperentitiesIndices( tnlDimensionsTag< Dimensions >() );
   }

   template< int Dimensions >
      const typename SuperentitiesTraits< Dimensions >::SharedContainerType& getSuperentitiesIndices() const
   {
      static_assert( SuperentitiesTraits< Dimensions >::available, "You try to get superentities which are not configured for storage." );
      typedef tnlMeshSuperentityAccess< MeshConfig, EntityTag >  SuperentityBaseType;
      return SuperentityBaseType::getSubentitiesIndices( tnlDimensionsTag< Dimensions >() );
   }

   /****
    * Vertices
    */
   enum { verticesCount = SubentitiesTraits< 0 >::subentitiesCount };
   typedef typename SubentitiesTraits< 0 >::ContainerType        ContainerType;
   typedef typename SubentitiesTraits< 0 >::SharedContainerType  SharedContainerType;
   typedef typename SubentitiesTraits< 0 >::GlobalIndexType      GlobalIndexType;
   typedef typename SubentitiesTraits< 0 >::LocalIndexType       LocalIndexType;

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
      static const LocalIndexType subentitiesCount = tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTag, tnlDimensionsTag<dim>>::count;
      tnlAssert( 0 <= index && index < subentitiesCount, );
      
      return SubentityStorageLayers::subentityOrientation( tnlDimensionsTag< dim >(), index );
   }  
   
   // TODO: This is only for the mesh initializer, fix this
   typedef tnlMeshSuperentityAccess< MeshConfig, EntityTag >                     SuperentityAccessBase;
   typedef typename tnlMeshTraits< MeshConfig>::IdArrayAccessorType        IdArrayAccessorType;
   typedef tnlMeshSubentityStorageLayers< MeshConfig, EntityTag >                SubentityStorageLayers;
   
   template< typename DimensionsTag >
   typename tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTag, DimensionsTag::value >::IdArrayType& subentityIdsArray()
   {
      return SubentityStorageLayers::subentityIdsArray( DimensionsTag() );
   }
   
   template<typename DimensionsTag >
   IdArrayAccessorType& superentityIdsArray()
   {
      return SuperentityAccessBase::superentityIdsArray( DimensionsTag());
   }
   
   template< typename DimensionsTag >
   typename tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTag, DimensionsTag::value >::OrientationArrayType& subentityOrientationsArray()
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

      /****
       * The entity typedefs
       */
      typedef MeshConfig         MeshMeshConfig;
      typedef tnlMeshVertexTopology  Tag;
      typedef tnlMeshEntitySeed< MeshConfig, tnlMeshVertexTopology >            SeedType;
      typedef typename tnlMeshTraits< MeshConfig >::PointType PointType;
      enum { dimensions = Tag::dimensions };
      enum { meshDimensions = tnlMeshTraits< MeshConfig >::meshDimensions };

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
                           //EntityTag::getType() + ", " +
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
   template< int Dimensions >
   struct SuperentitiesTraits
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
   };
   
   /*template< int Dimensions >
   bool setNumberOfSuperentities( const typename SuperentitiesTraits< Dimensions >::LocalIndexType size )
   {
      tnlAssert( size >= 0,
                 cerr << "size = " << size << endl; );
      typedef tnlMeshSuperentityStorageLayers< MeshConfig, tnlMeshVertexTopology >  SuperentityBaseType;
      return SuperentityBaseType::setNumberOfSuperentities( tnlDimensionsTag< Dimensions >(),
                                                            size );
   }*/

   template< int Dimensions >
   typename SuperentitiesTraits< Dimensions >::LocalIndexType getNumberOfSuperentities() const
   {
      typedef tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology >  SuperentityBaseType;
      //return SuperentityBaseType::getNumberOfSuperentities( tnlDimensionsTag< Dimensions >() );
   }

   template< int Dimensions >
      typename SuperentitiesTraits< Dimensions >::SharedContainerType& getSuperentitiesIndices()
   {
      typedef tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology >  SuperentityBaseType;
      //return SuperentityBaseType::getSuperentitiesIndices( tnlDimensionsTag< Dimensions >() );
   }

   template< int Dimensions >
      const typename SuperentitiesTraits< Dimensions >::SharedContainerType& getSuperentitiesIndeces() const
   {
      typedef tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology >  SuperentityBaseType;
      //return SuperentityBaseType::getSubentitiesIndices( tnlDimensionsTag< Dimensions >() );
   }

   /*template< int Dimensions >
   void setSuperentityIndex( const typename SuperentitiesTraits< Dimensions >::LocalIndexType localIndex,
                             const typename SuperentitiesTraits< Dimensions >::GlobalIndexType globalIndex )
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
   typename SuperentitiesTraits< Dimensions >::GlobalIndexType
      getSuperentityIndex( const typename SuperentitiesTraits< Dimensions >::LocalIndexType localIndex ) const
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
          typename EntityTag >
ostream& operator <<( ostream& str, const tnlMeshEntity< MeshConfig, EntityTag >& entity )
{
   entity.print( str );
   return str;
}

/****
 * This tells the compiler that theMeshEntity is a type with a dynamic memory allocation.
 * It is necessary for the loading and the saving of the mesh entities arrays.
 */
template< typename MeshConfig,
          typename EntityTag >
struct tnlDynamicTypeTag< tnlMeshEntity< MeshConfig, EntityTag > >
{
   enum { value = true };
};

#endif /* TNLMESHENTITY_H_ */
