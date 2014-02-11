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

#include <mesh/traits/tnlMeshTrait.h>

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntity
{
};

template< typename ConfigTag >
class tnlMeshEntity< ConfigTag, tnlMeshVertexTag >
{
   public:

   typedef ConfigTag  MeshConfigTag;
   typedef VertexTag  Tag;

   //typedef typename SuperentityBaseType::SharedArrayType  SuperentityIndicesArrayType;
   typedef typename tnlMeshTag< ConfigTag >::PointType PointType;

   enum { dimension = Tag::dimension };
   enum { meshDimension = tnlMeshTag< ConfigTag >::dimension };

   template< Dimensions >
   struct SuperentitiesAvailable
   {
      //enum { value = SuperentityStorage< ConfigTag, Tag, Dimensions >::enabled };
   };

   //using SuperentityBaseType::superentityIndices;
   //template< int Dimensions >
   //SuperentityIndicesArrayType superentityIndices() const { return this->superentityIndices(DimTag<dim>()); }

   PointType getPoint() const { return m_point; }

   void setPoint( const PointType& point ) { m_point = point; }

   protected:

   PointType point;
};


#endif /* TNLMESHENTITY_H_ */
