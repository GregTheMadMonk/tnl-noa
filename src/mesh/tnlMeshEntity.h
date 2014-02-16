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

#include <mesh/traits/tnlMeshTraits.h>
#include <mesh/traits/tnlDimensionsTraits.h>
#include <mesh/topologies/tnlMeshVertexTag.h>
#include <mesh/layers/tnlMeshSubentityStorageLayer.h>
#include <mesh/layers/tnlMeshSuperentityStorageLayer.h>

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntity
   : public tnlMeshSubentityStorageLayers< ConfigTag, EntityTag >,
     public tnlMeshSuperentityStorageLayers< ConfigTag, EntityTag >
{
   public:

   /****
    * Entity typedefs
    */
   typedef ConfigTag                                            MeshConfigTag;
   typedef EntityTag                                            Tag;
   enum { dimensions = Tag::dimensions };
   enum { meshDimensions = tnlMeshTraits< ConfigTag >::meshDimensions };

   /****
    * Subentities
    */
   template< int Dimensions >
   struct SubentitiesTraits
   {
      typedef tnlDimensionsTraits< Dimensions >                 DimensionsTraits;
      typedef tnlMeshSubentitiesTraits< ConfigTag,
                                        EntityTag,
                                        DimensionsTraits >      SubentityTraits;
      typedef typename SubentityTraits::ContainerType           ContainerType;
      typedef typename ContainerType::ElementType               GlobalIndexType;
      typedef int                                               LocalIndexType;
      // TODO: make this as:
      // typedef typename Type::IndexType   LocalIndexType
      enum { available = tnlMeshSubentityStorage< ConfigTag,
                                                  EntityTag,
                                                  Dimensions >::enabled };
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
      tnlAssert( 0 <= localIndex &&
                 localIndex < SubentitiesTraits< Dimensions >::subentitiesCount,
                 cerr << "localIndex = " << localIndex
                      << " subentitiesCount = "
                      << SubentitiesTraits< Dimensions >::subentitiesCount );
      typedef tnlMeshSubentityStorageLayers< ConfigTag, EntityTag >  SubentityBaseType;
      SubentityBaseType::setSubentityIndex( tnlDimensionsTraits< Dimensions >(),
                                            localIndex,
                                            globalIndex );
   }

   template< int Dimensions >
   typename SubentitiesTraits< Dimensions >::GlobalIndexType
      getSubentityIndex( const typename SubentitiesTraits< Dimensions >::LocalIndexType localIndex) const
      {
         tnlAssert( 0 <= localIndex &&
                    localIndex < SubentitiesTraits< Dimensions >::subentitiesCount,
                    cerr << "localIndex = " << localIndex
                         << " subentitiesCount = "
                         << SubentitiesTraits< Dimensions >::subentitiesCount );
         typedef tnlMeshSubentityStorageLayers< ConfigTag, EntityTag >  SubentityBaseType;
         return SubentityBaseType::getSubentityIndex( tnlDimensionsTraits< Dimensions >(),
                                                      localIndex );
      }

   /****
    * Superentities
    */
   template< int Dimensions >
   struct SuperentitiesTraits
   {
      typedef tnlDimensionsTraits< Dimensions >                 DimensionsTraits;
      typedef tnlMeshSuperentitiesTraits< ConfigTag,
                                          EntityTag,
                                          DimensionsTraits >    SuperentityTraits;
      typedef typename SuperentityTraits::ContainerType         ContainerType;
      typedef typename ContainerType::ElementType               GlobalIndexType;
      typedef int                                               LocalIndexType;
      // TODO: make this as:
      // typedef typename Type::IndexType   LocalIndexType
      enum { available = tnlMeshSuperentityStorage< ConfigTag,
                                                    EntityTag,
                                                    Dimensions >::enabled };
   };

   template< int Dimensions >
   bool setNumberOfSuperentities( const typename SuperentitiesTraits< Dimensions >::LocalIndexType size )
   {
      tnlAssert( size >= 0,
                 cerr << "size = " << size << endl; );
      typedef tnlMeshSuperentityStorageLayers< ConfigTag, EntityTag >  SuperentityBaseType;
      return SuperentityBaseType::setNumberOfSuperentities( tnlDimensionsTraits< Dimensions >(),
                                                            size );
   }

   template< int Dimensions >
   typename SuperentitiesTraits< Dimensions >::LocalIndexType getNumberOfSuperentities() const
   {
      typedef tnlMeshSuperentityStorageLayers< ConfigTag, EntityTag >  SuperentityBaseType;
      return SuperentityBaseType::getNumberOfSuperentities( tnlDimensionsTraits< Dimensions >() );
   }

   template< int Dimensions >
   void setSuperentityIndex( const typename SuperentitiesTraits< Dimensions >::LocalIndexType localIndex,
                             const typename SuperentitiesTraits< Dimensions >::GlobalIndexType globalIndex )
   {
      tnlAssert( localIndex < this->getNumberOfSuperentities< Dimensions >(),
                 cerr << " localIndex = " << localIndex
                      << " this->getNumberOfSuperentities< Dimensions >() = " << this->getNumberOfSuperentities< Dimensions >() << endl; );
      typedef tnlMeshSuperentityStorageLayers< ConfigTag, EntityTag >  SuperentityBaseType;
      SuperentityBaseType::setSuperentityIndex( tnlDimensionsTraits< Dimensions >(),
                                                localIndex,
                                                globalIndex );
   }

   template< int Dimensions >
   typename SuperentitiesTraits< Dimensions >::GlobalIndexType 
      getSuperentityIndex( const typename SuperentitiesTraits< Dimensions >::LocalIndexType localIndex ) const
   {
      tnlAssert( localIndex < this->getNumberOfSuperentities< Dimensions >(),
                 cerr << " localIndex = " << localIndex
                      << " this->getNumberOfSuperentities< Dimensions >() = " << this->getNumberOfSuperentities< Dimensions >() << endl; );
      typedef tnlMeshSuperentityStorageLayers< ConfigTag, EntityTag >  SuperentityBaseType;
      return SuperentityBaseType::getSuperentityIndex( tnlDimensionsTraits< Dimensions >(),
                                                       localIndex );
   }

   /****
    * Vertices
    */
   enum { verticesCount = SubentitiesTraits< 0 >::subentitiesCount };
   typedef typename SubentitiesTraits< 0 >::GlobalIndexType  VerticesGlobalIndexType;
   typedef typename SubentitiesTraits< 0 >::LocalIndexType   VerticesLocalIndexType;

   VerticesLocalIndexType getNumberOfVertices() const
   {
      return verticesCount;
   }

   void setVertexIndex( const VerticesLocalIndexType localIndex,
                        const VerticesGlobalIndexType globalIndex )
   {
      this->setSubentityIndex< 0 >( localIndex, globalIndex  );
   }

   VerticesGlobalIndexType getVertexIndex( const VerticesLocalIndexType localIndex )
   {
      return this->getSubentityIndex< 0 >( localIndex  );
   }
};

template< typename ConfigTag >
class tnlMeshEntity< ConfigTag, tnlMeshVertexTag >
   : public tnlMeshSuperentityStorageLayers< ConfigTag, tnlMeshVertexTag >
{
   public:

   /****
    * The entity typedefs
    */
   typedef ConfigTag         MeshConfigTag;
   typedef tnlMeshVertexTag  Tag;
   typedef typename tnlMeshTraits< ConfigTag >::PointType PointType;
   enum { dimensions = Tag::dimensions };
   enum { meshDimensions = tnlMeshTraits< ConfigTag >::meshDimensions };

   /****
    * Superentities
    */
   template< int Dimensions >
   struct SuperentitiesTraits
   {
      typedef tnlDimensionsTraits< Dimensions >                 DimensionsTraits;
      typedef tnlMeshSuperentitiesTraits< ConfigTag,
                                          tnlMeshVertexTag,
                                          DimensionsTraits >    SuperentityTraits;
      typedef typename SuperentityTraits::ContainerType         ContainerType;
      typedef typename ContainerType::ElementType               GlobalIndexType;
      typedef int                                               LocalIndexType;
      // TODO: make this as:
      // typedef typename Type::IndexType   LocalIndexType
      enum { available = tnlMeshSuperentityStorage< ConfigTag,
                                                    tnlMeshVertexTag,
                                                    Dimensions >::enabled };
   };
   template< int Dimensions >
   bool setNumberOfSuperentities( const typename SuperentitiesTraits< Dimensions >::LocalIndexType size )
   {
      tnlAssert( size >= 0,
                 cerr << "size = " << size << endl; );
      typedef tnlMeshSuperentityStorageLayers< ConfigTag, tnlMeshVertexTag >  SuperentityBaseType;
      return SuperentityBaseType::setNumberOfSuperentities( tnlDimensionsTraits< Dimensions >(),
                                                            size );
   }

   template< int Dimensions >
   typename SuperentitiesTraits< Dimensions >::LocalIndexType getNumberOfSuperentities() const
   {
      typedef tnlMeshSuperentityStorageLayers< ConfigTag, tnlMeshVertexTag >  SuperentityBaseType;
      return SuperentityBaseType::getNumberOfSuperentities( tnlDimensionsTraits< Dimensions >() );
   }

   template< int Dimensions >
   void setSuperentityIndex( const typename SuperentitiesTraits< Dimensions >::LocalIndexType localIndex,
                             const typename SuperentitiesTraits< Dimensions >::GlobalIndexType globalIndex )
   {
      tnlAssert( localIndex < this->getNumberOfSuperentities< Dimensions >(),
                 cerr << " localIndex = " << localIndex
                      << " this->getNumberOfSuperentities< Dimensions >() = " << this->getNumberOfSuperentities< Dimensions >() << endl; );
      typedef tnlMeshSuperentityStorageLayers< ConfigTag, tnlMeshVertexTag >  SuperentityBaseType;
      SuperentityBaseType::setSuperentityIndex( tnlDimensionsTraits< Dimensions >(),
                                                localIndex,
                                                globalIndex );
   }

   template< int Dimensions >
   typename SuperentitiesTraits< Dimensions >::GlobalIndexType
      getSuperentityIndex( const typename SuperentitiesTraits< Dimensions >::LocalIndexType localIndex ) const
   {
      tnlAssert( localIndex < this->getNumberOfSuperentities< Dimensions >(),
                 cerr << " localIndex = " << localIndex
                      << " this->getNumberOfSuperentities< Dimensions >() = " << this->getNumberOfSuperentities< Dimensions >() << endl; );
      typedef tnlMeshSuperentityStorageLayers< ConfigTag, tnlMeshVertexTag >  SuperentityBaseType;
      return SuperentityBaseType::getSuperentityIndex( tnlDimensionsTraits< Dimensions >(),
                                                       localIndex );
   }

   /****
    * Points
    */
   PointType getPoint() const { return this->point; }

   void setPoint( const PointType& point ) { this->point = point; }

   protected:

   PointType point;
};


#endif /* TNLMESHENTITY_H_ */
