/***************************************************************************
                          tnlMeshSubentityStorageLayer.h  -  description
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

#ifndef TNLMESHSUBENTITYSTORAGELAYER_H_
#define TNLMESHSUBENTITYSTORAGELAYER_H_

#include <mesh/traits/tnlDimensionsTag.h>
#include <mesh/traits/tnlStorageTag.h>

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionTag,
          typename SubentityStorageTag = typename SubentitiesTag< ConfigTag,
                                                                  EntityTag,
                                                                  DimensionTag >::SubentityStorageTag >
class tnlMeshSubentityStorageLayer;


template< typename ConfigTag,
          typename EntityTag >
class tnlMeshSubentityStorageLayers
   : public tnlMeshSubentityStorageLayer< ConfigTag,
                                          EntityTag,
                                          tnlDimensionsTag< EntityTag::dimension - 1 > >
{
};


template< typename ConfigTag,
          typename EntityTag,
          typename DimensionTag >
class tnlMeshSubentityStorageLayer< ConfigTag,
                                    EntityTag,
                                    DimensionTag,
                                    tnlStorageTag< true > >
   : public tnlMeshSubentityStorageLayer< ConfigTag,
                                          EntityTag,
                                          typename DimensionTag::Previous >
{
   typedef tnlMeshSubentityStorageLayer< ConfigTag,
                                         EntityTag,
                                         typename DimensionTag::Previous > BaseType;

   typedef tnlMeshSubentitiesTag< ConfigTag,
                                  EntityTag,
                                  DimensionTag> SubentityTag;

   protected:

   typedef typename tnlMeshSubentityTag::ContainerType    ContainerType;
   typedef typename tnlMeshSubentityTag::SharedArrayType  SharedArrayType;

   using BaseType::subentityIndices;
   SharedArrayType subentityIndices(DimensionTag) const   { return SharedArrayType(m_subentityEntities); }

   using BaseType::subentityIndicesContainer;
   ContainerType &subentityIndicesContainer(DimensionTag) { return m_subentityEntities; }

private:
   ContainerType m_subentityEntities;
};


template<typename ConfigTag, typename EntityTag, typename DimensionTag>
class SubentityStorageLayer<ConfigTag, EntityTag, DimensionTag, StorageTag<false> >
        : public SubentityStorageLayer<ConfigTag, EntityTag, typename DimensionTag::Previous>
{
};


template<typename ConfigTag, typename EntityTag>
class SubentityStorageLayer<ConfigTag, EntityTag, DimTag<0>, StorageTag<true> >
{
   typedef DimTag<0>                                          DimensionTag;

   typedef SubentitiesTag<ConfigTag, EntityTag, DimensionTag> SubentityTag;

protected:
   typedef typename SubentityTag::ContainerType               ContainerType;
   typedef typename SubentityTag::SharedArrayType             SharedArrayType;

   SharedArrayType subentityIndices(DimensionTag) const   { return SharedArrayType(m_subentityVertices); }

   ContainerType &subentityIndicesContainer(DimensionTag) { return m_subentityVertices; }

private:
   ContainerType m_subentityVertices;
};




#endif /* TNLMESHSUBENTITYSTORAGELAYER_H_ */
