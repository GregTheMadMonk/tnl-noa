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

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntity
   : public tnlMeshSubentityStorageLayers< ConfigTag, EntityTag >
{
   public:

   typedef tnlMeshSubentityStorageLayers< ConfigTag, 
                                          EntityTag >            SubentityBaseType;
   typedef tnlMeshSubentitiesTraits< ConfigTag, 
                                     EntityTag,
                                     tnlDimensionsTraits< 0 > >  SubvertexTraits;
   typedef typename SubvertexTraits::ContainerType::ElementType  GlobalIndexType;
   //typedef typename SubvertexTraits::ContainerType::IndexType   LocalIndexType;
   // TODO: the above requires IndexType in tnlStaticArray - think about it
   typedef  int                                                   LocalIndexType;

   enum { subverticesCount = SubvertexTraits::count };

   public:

   typedef ConfigTag                                        MeshConfigTag;
   typedef EntityTag                                        Tag;

   typedef typename SubentityBaseType::SharedArrayType      SubentityIndicesArrayType;
   typedef typename SubentityBaseType::ContainerType        SubentityContainerType;
   //typedef typename SuperentityBaseType::SharedArrayType    SuperentityIndicesArrayType;

   enum { dimensions = Tag::dimensions };
   enum { meshDimensions = tnlMeshTraits< ConfigTag >::meshDimensions };

   template< int Dimensions >
   struct SubentitiesAvailable
   {
      enum { value = tnlMeshSubentityStorage< ConfigTag,
                                              EntityTag,
                                              Dimensions >::enabled };
   };

   template< int Dimensions >
   struct SuperentitiesAvailable
   {
      /*enum { value = tnlSuperentityStorage< ConfigTag,
                                            EntityTag,
                                            Dimensions >::enabled };*/
   };

   using SubentityBaseType::getSubentityIndices;
   template< int Dimensions >
   SubentityIndicesArrayType getSubentityIndices() const
      { return this->getSubentityIndices( tnlDimensionsTraits< Dimensions >() ); }

   template< int Dimensions >
   SubentityContainerType& getSubentityIndices()
      {
         return this->getSubentityIndices( tnlDimensionsTraits< Dimensions >() );
      }

   /*using SuperentityBaseType::superentityIndices;
   template<DimensionType dim>
   SuperentityIndicesArrayType superentityIndices() const { return this->superentityIndices(DimTag<dim>()); }
   */

   void setVertexIndex( const LocalIndexType localIndex,
                        const GlobalIndexType globalIndex )
   {
      tnlAssert( 0 <= localIndex && localIndex < subverticesCount,
                cerr << "localIndex = " << localIndex
                     << " subverticesCount = " << subverticesCount );

      this->getSubentityIndices( tnlDimensionsTraits< 0 >() )[ localIndex ] = globalIndex;
   }

   GlobalIndexType getVertexIndex( const LocalIndexType localIndex )
   {
      tnlAssert( 0 <= localIndex && localIndex < subverticesCount,
                 cerr << "localIndex = " << localIndex
                      << " subverticesCount = " << subverticesCount );
      return this->getSubentityIndices( tnlDimensionsTraits< 0 >() )[ localIndex ];
   }
};

template< typename ConfigTag >
class tnlMeshEntity< ConfigTag, tnlMeshVertexTag >
{
   public:

   typedef ConfigTag         MeshConfigTag;
   typedef tnlMeshVertexTag  Tag;

   //typedef typename SuperentityBaseType::SharedArrayType  SuperentityIndicesArrayType;
   typedef typename tnlMeshTraits< ConfigTag >::PointType PointType;

   enum { dimensions = Tag::dimensions };
   enum { meshDimensions = tnlMeshTraits< ConfigTag >::meshDimensions };

   template< int Dimensions >
   struct SuperentitiesAvailable
   {
      //enum { value = SuperentityStorage< ConfigTag, Tag, Dimensions >::enabled };
   };

   //using SuperentityBaseType::superentityIndices;
   //template< int Dimensions >
   //SuperentityIndicesArrayType superentityIndices() const { return this->superentityIndices(DimTag<dim>()); }

   PointType getPoint() const { return this->point; }

   void setPoint( const PointType& point ) { this->point = point; }

   protected:

   PointType point;
};


#endif /* TNLMESHENTITY_H_ */
