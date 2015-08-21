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
#include <mesh/topologies/tnlMeshVertexTag.h>
#include <mesh/layers/tnlMeshSubentityStorageLayer.h>
#include <mesh/layers/tnlMeshSuperentityStorageLayer.h>
#include <mesh/layers/tnlMeshSuperentityAccess.h>
#include <mesh/tnlMeshEntitySeed.h>

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntity
   : public tnlMeshSubentityStorageLayers< ConfigTag, EntityTag >,     
     public tnlMeshSuperentityAccess< ConfigTag, EntityTag >,
     public tnlMeshEntityId< typename ConfigTag::IdType,
                             typename ConfigTag::GlobalIndexType >
{
   public:
      
      // TODO: This is only because of STD lib bug in tnlIndexedSet
      tnlMeshEntity( const tnlMeshEntity& entyti ) {}
      tnlMeshEntity() {}

   static tnlString getType()
   {
      return tnlString( "tnlMesh< " ) +
                        //ConfigTag::getType() + ", " +
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
      if( ! tnlMeshSubentityStorageLayers< ConfigTag, EntityTag >::save( file ) /*||
          ! tnlMeshSuperentityStorageLayers< ConfigTag, EntityTag >::save( file )*/ )
         return false;
      return true;
   }

   bool load( tnlFile& file )
   {
      if( ! tnlMeshSubentityStorageLayers< ConfigTag, EntityTag >::load( file ) /*||
          ! tnlMeshSuperentityStorageLayers< ConfigTag, EntityTag >::load( file ) */ )
         return false;
      return true;
   }

   void print( ostream& str ) const
   {
      str << "\t Mesh entity dimensions: " << EntityTag::dimensions << endl;
      tnlMeshSubentityStorageLayers< ConfigTag, EntityTag >::print( str );
      tnlMeshSuperentityAccess< ConfigTag, EntityTag >::print( str );
   }

   bool operator==( const tnlMeshEntity& entity ) const
   {
      return ( tnlMeshSubentityStorageLayers< ConfigTag, EntityTag >::operator==( entity ) &&
               tnlMeshSuperentityAccess< ConfigTag, EntityTag >::operator==( entity ) &&
               tnlMeshEntityId< typename ConfigTag::IdType,
                                typename ConfigTag::GlobalIndexType >::operator==( entity ) );
   }


   /****
    * Entity typedefs
    */
   typedef ConfigTag                                            MeshConfigTag;
   typedef EntityTag                                            Tag;
   typedef tnlMeshEntitySeed< ConfigTag, EntityTag >            SeedType;
   enum { dimensions = Tag::dimensions };
   enum { meshDimensions = tnlMeshTraits< ConfigTag >::meshDimensions };

   /****
    * Subentities
    */
   template< int Dimensions >
   struct SubentitiesTraits
   {
      static_assert( Dimensions < meshDimensions, "Asking for subentities with more or the same number of dimensions then the mesh itself." );
      typedef tnlDimensionsTag< Dimensions >                 DimensionsTag;
      typedef tnlMeshSubentitiesTraits< ConfigTag,
                                        EntityTag,
                                        DimensionsTag >      SubentityTraits;
      typedef typename SubentityTraits::ContainerType           ContainerType;
      typedef typename SubentityTraits::SharedContainerType     SharedContainerType;
      typedef typename ContainerType::ElementType               GlobalIndexType;
      typedef int                                               LocalIndexType;

      // TODO: make this as:
      // typedef typename Type::IndexType   LocalIndexType
      /*enum { available = tnlMeshSubentityStorage< ConfigTag,
                                                  EntityTag,
                                                  Dimensions >::enabled };*/
      static const bool available = ConfigTag::template subentityStorage( EntityTag(), Dimensions );
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
      typedef tnlMeshSubentityStorageLayers< ConfigTag, EntityTag >  SubentityBaseType;
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
         typedef tnlMeshSubentityStorageLayers< ConfigTag, EntityTag >  SubentityBaseType;
         return SubentityBaseType::getSubentityIndex( tnlDimensionsTag< Dimensions >(),
                                                      localIndex );
      }

   template< int Dimensions >
      typename SubentitiesTraits< Dimensions >::SharedContainerType&
         getSubentitiesIndices()
   {
      static_assert( SubentitiesTraits< Dimensions >::available, "You try to get subentities which are not configured for storage." );
      typedef tnlMeshSubentityStorageLayers< ConfigTag, EntityTag >  SubentityBaseType;
      return SubentityBaseType::getSubentitiesIndices( tnlDimensionsTag< Dimensions >() );
   }

   template< int Dimensions >
      const typename SubentitiesTraits< Dimensions >::SharedContainerType&
         getSubentitiesIndices() const
   {
      static_assert( SubentitiesTraits< Dimensions >::available, "You try to set subentities which are not configured for storage." );
      typedef tnlMeshSubentityStorageLayers< ConfigTag, EntityTag >  SubentityBaseType;
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
      typedef tnlMeshSuperentitiesTraits< ConfigTag,
                                          EntityTag,
                                          DimensionsTag >    SuperentityTraits;
      typedef typename SuperentityTraits::ContainerType         ContainerType;
      typedef typename SuperentityTraits::SharedContainerType   SharedContainerType;
      typedef typename ContainerType::ElementType               GlobalIndexType;
      typedef int                                               LocalIndexType;
      // TODO: make this as:
      // typedef typename Type::IndexType   LocalIndexType
      static const bool available = ConfigTag::template superentityStorage( EntityTag(), Dimensions );
   };

   /*template< int Dimensions >
   bool setNumberOfSuperentities( const typename SuperentitiesTraits< Dimensions >::LocalIndexType size )
   {
      static_assert( SuperentitiesTraits< Dimensions >::available, "You try to set number of superentities which are not configured for storage." );
      tnlAssert( size >= 0,
                 cerr << "size = " << size << endl; );
      typedef tnlMeshSuperentityStorageLayers< ConfigTag, EntityTag >  SuperentityBaseType;
      return SuperentityBaseType::setNumberOfSuperentities( tnlDimensionsTag< Dimensions >(),
                                                            size );
   }*/

   template< int Dimensions >
   typename SuperentitiesTraits< Dimensions >::LocalIndexType getNumberOfSuperentities() const
   {
      static_assert( SuperentitiesTraits< Dimensions >::available, "You try to get number of superentities which are not configured for storage." );
      typedef tnlMeshSuperentityAccess< ConfigTag, EntityTag >  SuperentityBaseType;
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
      typedef tnlMeshSuperentityAccess< ConfigTag, EntityTag >  SuperentityBaseType;
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
      typedef tnlMeshSuperentityAccess< ConfigTag, EntityTag >  SuperentityBaseType;
      return SuperentityBaseType::getSuperentityIndex( tnlDimensionsTag< Dimensions >(),
                                                       localIndex );
   }

   template< int Dimensions >
      typename SuperentitiesTraits< Dimensions >::SharedContainerType& getSuperentitiesIndices()
   {
      static_assert( SuperentitiesTraits< Dimensions >::available, "You try to get superentities which are not configured for storage." );
      typedef tnlMeshSuperentityAccess< ConfigTag, EntityTag >  SuperentityBaseType;
      //return SuperentityBaseType::getSuperentitiesIndices( tnlDimensionsTag< Dimensions >() );
   }

   template< int Dimensions >
      const typename SuperentitiesTraits< Dimensions >::SharedContainerType& getSuperentitiesIndices() const
   {
      static_assert( SuperentitiesTraits< Dimensions >::available, "You try to get superentities which are not configured for storage." );
      typedef tnlMeshSuperentityAccess< ConfigTag, EntityTag >  SuperentityBaseType;
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
   
   
   // TODO: This is only for the mesh initializer, fix this
   typedef tnlMeshSuperentityAccess< ConfigTag, EntityTag >                     SuperentityAccessBase;
   typedef typename tnlMeshConfigTraits< ConfigTag>::IdArrayAccessorType        IdArrayAccessorType;
   
   template<typename DimensionsTag >
	IdArrayAccessorType& superentityIdsArray()
	{
		return SuperentityAccessBase::superentityIdsArray( DimensionsTag());
	}
};

template< typename ConfigTag >
class tnlMeshEntity< ConfigTag, tnlMeshVertexTag >
   : public tnlMeshSuperentityAccess< ConfigTag, tnlMeshVertexTag >,
     public tnlMeshEntityId< typename ConfigTag::IdType,
                             typename ConfigTag::GlobalIndexType >
{
   public:

      // TODO: This is only because of STD lib bug in tnlIndexedSet
      tnlMeshEntity( const tnlMeshEntity& entyti ) {}
      tnlMeshEntity() {}

      
   static tnlString getType()
   {
      return tnlString( "tnlMesh< " ) +
                        //ConfigTag::getType() + ", " +
                        //EntityTag::getType() + ", " +
                        " >";
   }

   tnlString getTypeVirtual() const
   {
      return this->getType();
   }

   /****
    * The entity typedefs
    */
   typedef ConfigTag         MeshConfigTag;
   typedef tnlMeshVertexTag  Tag;
   typedef tnlMeshEntitySeed< ConfigTag, tnlMeshVertexTag >            SeedType;
   typedef typename tnlMeshTraits< ConfigTag >::PointType PointType;
   enum { dimensions = Tag::dimensions };
   enum { meshDimensions = tnlMeshTraits< ConfigTag >::meshDimensions };

   /*~tnlMeshEntity()
   {
      cerr << "   Destroying entity with " << tnlMeshVertexTag::dimensions << " dimensions..." << endl;
   }*/

   bool save( tnlFile& file ) const
   {
      if( //! tnlMeshSuperentityStorageLayers< ConfigTag, tnlMeshVertexTag >::save( file ) ||
          ! point.save( file ) )
         return false;
      return true;
   }

   bool load( tnlFile& file )
   {
      if( //! tnlMeshSuperentityStorageLayers< ConfigTag, tnlMeshVertexTag >::load( file ) ||
          ! point.load( file ) )
         return false;
      return true;
   }

   void print( ostream& str ) const
   {
      str << "\t Mesh entity dimensions: " << tnlMeshVertexTag::dimensions << endl;
      str << "\t Coordinates = ( " << point << " )";
      tnlMeshSuperentityAccess< ConfigTag, tnlMeshVertexTag >::print( str );
   }

   bool operator==( const tnlMeshEntity& entity ) const
   {
      return ( //tnlMeshSuperentityAccess< ConfigTag, tnlMeshVertexTag >::operator==( entity ) &&
               tnlMeshEntityId< typename ConfigTag::IdType,
                                typename ConfigTag::GlobalIndexType >::operator==( entity ) &&
               point == entity.point );

   }

   /****
    * Superentities
    */
   template< int Dimensions >
   struct SuperentitiesTraits
   {
      typedef tnlDimensionsTag< Dimensions >                 DimensionsTag;
      typedef tnlMeshSuperentitiesTraits< ConfigTag,
                                          tnlMeshVertexTag,
                                          DimensionsTag >    SuperentityTraits;
      typedef typename SuperentityTraits::ContainerType         ContainerType;
      typedef typename SuperentityTraits::SharedContainerType   SharedContainerType;
      typedef typename ContainerType::ElementType               GlobalIndexType;
      typedef int                                               LocalIndexType;
      // TODO: make this as:
      // typedef typename Type::IndexType   LocalIndexType
      /*enum { available = tnlMeshSuperentityStorage< ConfigTag,
                                                    tnlMeshVertexTag,
                                                    Dimensions >::enabled };*/
      static const bool available = ConfigTag::template superentityStorage< tnlMeshVertexTag >( Dimensions );
   };
   
   /*template< int Dimensions >
   bool setNumberOfSuperentities( const typename SuperentitiesTraits< Dimensions >::LocalIndexType size )
   {
      tnlAssert( size >= 0,
                 cerr << "size = " << size << endl; );
      typedef tnlMeshSuperentityStorageLayers< ConfigTag, tnlMeshVertexTag >  SuperentityBaseType;
      return SuperentityBaseType::setNumberOfSuperentities( tnlDimensionsTag< Dimensions >(),
                                                            size );
   }*/

   template< int Dimensions >
   typename SuperentitiesTraits< Dimensions >::LocalIndexType getNumberOfSuperentities() const
   {
      typedef tnlMeshSuperentityAccess< ConfigTag, tnlMeshVertexTag >  SuperentityBaseType;
      //return SuperentityBaseType::getNumberOfSuperentities( tnlDimensionsTag< Dimensions >() );
   }

   template< int Dimensions >
      typename SuperentitiesTraits< Dimensions >::SharedContainerType& getSuperentitiesIndices()
   {
      typedef tnlMeshSuperentityAccess< ConfigTag, tnlMeshVertexTag >  SuperentityBaseType;
      //return SuperentityBaseType::getSuperentitiesIndices( tnlDimensionsTag< Dimensions >() );
   }

   template< int Dimensions >
      const typename SuperentitiesTraits< Dimensions >::SharedContainerType& getSuperentitiesIndeces() const
   {
      typedef tnlMeshSuperentityAccess< ConfigTag, tnlMeshVertexTag >  SuperentityBaseType;
      //return SuperentityBaseType::getSubentitiesIndices( tnlDimensionsTag< Dimensions >() );
   }

   /*template< int Dimensions >
   void setSuperentityIndex( const typename SuperentitiesTraits< Dimensions >::LocalIndexType localIndex,
                             const typename SuperentitiesTraits< Dimensions >::GlobalIndexType globalIndex )
   {
      tnlAssert( localIndex < this->getNumberOfSuperentities< Dimensions >(),
                 cerr << " localIndex = " << localIndex
                      << " this->getNumberOfSuperentities< Dimensions >() = " << this->getNumberOfSuperentities< Dimensions >() << endl; );
      typedef tnlMeshSuperentityAccess< ConfigTag, tnlMeshVertexTag >  SuperentityBaseType;
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
      typedef tnlMeshSuperentityAccess< ConfigTag, tnlMeshVertexTag >  SuperentityBaseType;
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
   typedef typename tnlMeshConfigTraits< ConfigTag>::IdArrayAccessorType        IdArrayAccessorType;
   typedef tnlMeshSuperentityAccess< ConfigTag, tnlMeshVertexTag > SuperentityAccessBase;
   
   template<typename DimensionsTag >
	IdArrayAccessorType& superentityIdsArray()
	{
		return SuperentityAccessBase::superentityIdsArray( DimensionsTag());
	}

};

template< typename ConfigTag,
          typename EntityTag >
ostream& operator <<( ostream& str, const tnlMeshEntity< ConfigTag, EntityTag >& entity )
{
   entity.print( str );
   return str;
}

/****
 * This tells the compiler that theMeshEntity is a type with a dynamic memory allocation.
 * It is necessary for the loading and the saving of the mesh entities arrays.
 */
template< typename ConfigTag,
          typename EntityTag >
struct tnlDynamicTypeTag< tnlMeshEntity< ConfigTag, EntityTag > >
{
   enum { value = true };
};

#endif /* TNLMESHENTITY_H_ */
