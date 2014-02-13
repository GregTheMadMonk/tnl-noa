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

   /****
    * The entity typedefs
    */
   typedef ConfigTag                                        MeshConfigTag;
   typedef EntityTag                                        Tag;
   enum { dimensions = Tag::dimensions };
   enum { meshDimensions = tnlMeshTraits< ConfigTag >::meshDimensions };

   // TODO: Global index type should be unique as well to make this simpler
   typedef  int                                                   LocalIndexType;

   /****
    * Vertices
    */
   typedef tnlMeshSubentityStorageLayer< ConfigTag,
                                         EntityTag,
                                         tnlDimensionsTraits< 0 > > VertexBaseType;
   typedef tnlMeshSubentitiesTraits< ConfigTag, 
                                     EntityTag,
                                     tnlDimensionsTraits< 0 > >  SubvertexTraits;
   typedef typename SubvertexTraits::ContainerType::ElementType  GlobalIndexType;
   //typedef typename SubvertexTraits::ContainerType::IndexType   LocalIndexType;
   // TODO: the above requires IndexType in tnlStaticArray - think about it


   enum { subverticesCount = SubvertexTraits::count };

   void setVertexIndex( const LocalIndexType localIndex,
                        const GlobalIndexType globalIndex )
   {
      tnlAssert( 0 <= localIndex && localIndex < subverticesCount,
                cerr << "localIndex = " << localIndex
                     << " subverticesCount = " << subverticesCount );

      VertexBaseType::setSubentityIndex( tnlDimensionsTraits< 0 >(),
                                         localIndex,
                                         globalIndex );
   }

   GlobalIndexType getVertexIndex( const LocalIndexType localIndex )
   {
      tnlAssert( 0 <= localIndex && localIndex < subverticesCount,
                 cerr << "localIndex = " << localIndex
                      << " subverticesCount = " << subverticesCount );
      return VertexBaseType::getSubentityIndex( tnlDimensionsTraits< 0 >(),
                                                localIndex );
   }

   /****
    * Subentities
    */
   template< int Dimensions >
   struct SubentityContainer
   {
      typedef tnlDimensionsTraits< Dimensions >  DimensionsTraits;
      typedef tnlMeshSubentitiesTraits< ConfigTag,
                                        EntityTag,
                                        DimensionsTraits > SubentityTraits;
      typedef typename SubentityTraits::ContainerType              Type;
      typedef typename Type::ElementType                           GlobalIndexType;
   };

   template< int Dimensions >
   struct SubentitiesAvailable
   {
      enum { value = tnlMeshSubentityStorage< ConfigTag,
                                              EntityTag,
                                              Dimensions >::enabled };
   };

   typedef tnlMeshSubentityStorageLayers< ConfigTag,
                                          EntityTag >            SubentityBaseType;

   template< int Dimensions >
   void setSubentityIndex( const LocalIndexType localIndex,
                           typename SubentityContainer< Dimensions >::GlobalIndexType globalIndex )
   {
      SubentityBaseType::setSubentityIndex( tnlDimensionsTraits< Dimensions >(),
                                            localIndex,
                                            globalIndex );
   }

   template< int Dimensions >
   typename SubentityContainer< Dimensions >::GlobalIndexType
      getSubentityIndex( const LocalIndexType localIndex) const
      {
         return SubentityBaseType::getSubentityIndex( tnlDimensionsTraits< Dimensions >(),
                                                      localIndex );
      }

   /****
    * Superentities containers
    */
   //typedef typename SuperentityBaseType::SharedArrayType    SuperentityIndicesArrayType;

   template< int Dimensions >
   struct SuperentitiesAvailable
   {
      /*enum { value = tnlSuperentityStorage< ConfigTag,
                                            EntityTag,
                                            Dimensions >::enabled };*/
   };



   /*using SuperentityBaseType::superentityIndices;
   template<DimensionType dim>
   SuperentityIndicesArrayType superentityIndices() const { return this->superentityIndices(DimTag<dim>()); }
   */


};

template< typename ConfigTag >
class tnlMeshEntity< ConfigTag, tnlMeshVertexTag >
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

   //typedef typename SuperentityBaseType::SharedArrayType  SuperentityIndicesArrayType;

   template< int Dimensions >
   struct SuperentitiesAvailable
   {
      //enum { value = SuperentityStorage< ConfigTag, Tag, Dimensions >::enabled };
   };

   //using SuperentityBaseType::superentityIndices;
   //template< int Dimensions >
   //SuperentityIndicesArrayType superentityIndices() const { return this->superentityIndices(DimTag<dim>()); }

   /****
    * Points
    */
   PointType getPoint() const { return this->point; }

   void setPoint( const PointType& point ) { this->point = point; }

   protected:

   PointType point;
};


#endif /* TNLMESHENTITY_H_ */
