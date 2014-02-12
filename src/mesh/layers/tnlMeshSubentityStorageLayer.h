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

#include <mesh/traits/tnlDimensionsTraits.h>
#include <mesh/traits/tnlStorageTraits.h>
#include <mesh/traits/tnlMeshSubentitiesTraits.h>

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionTag,
          typename SubentityStorageTag =
                   typename tnlMeshSubentitiesTraits< ConfigTag,
                                                      EntityTag,
                                                      DimensionTag >::SubentityStorageTag >
class tnlMeshSubentityStorageLayer;


template< typename ConfigTag,
          typename EntityTag >
class tnlMeshSubentityStorageLayers
   : public tnlMeshSubentityStorageLayer< ConfigTag,
                                          EntityTag,
                                          tnlDimensionsTraits< EntityTag::dimensions - 1 > >
{
};


template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTraits >
class tnlMeshSubentityStorageLayer< ConfigTag,
                                    EntityTag,
                                    DimensionsTraits,
                                    tnlStorageTraits< true > >
   : public tnlMeshSubentityStorageLayer< ConfigTag,
                                          EntityTag,
                                          typename DimensionsTraits::Previous >
{
   typedef tnlMeshSubentityStorageLayer< ConfigTag,
                                         EntityTag,
                                         typename DimensionsTraits::Previous > BaseType;

   typedef tnlMeshSubentitiesTraits< ConfigTag,
                                     EntityTag,
                                     DimensionsTraits > SubentityTraits;

   protected:

   typedef typename SubentityTraits::ContainerType    ContainerType;
   typedef typename SubentityTraits::SharedArrayType  SharedArrayType;

   using BaseType::getSubentityIndices;

   SharedArrayType getSubentityIndices( DimensionsTraits ) const
      { return SharedArrayType( subentityEntities); }

   //using BaseType::getSubentityIndicesContainer;
   ContainerType& getSubentityIndices( DimensionsTraits )
      { return this->subentityEntities; }

private:
   ContainerType subentityEntities;
};


template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTraits >
class tnlMeshSubentityStorageLayer< ConfigTag,
                                    EntityTag,
                                    DimensionsTraits,
                                    tnlStorageTraits< false > >
   : public tnlMeshSubentityStorageLayer< ConfigTag,
                                          EntityTag,
                                          typename DimensionsTraits::Previous >
{
};


template< typename ConfigTag,
          typename EntityTag >
class tnlMeshSubentityStorageLayer< ConfigTag,
                                    EntityTag,
                                    tnlDimensionsTraits< 0 >,
                                    tnlStorageTraits< true > >
{
   typedef tnlDimensionsTraits< 0 >                           DimensionsTraits;

   typedef tnlMeshSubentitiesTraits< ConfigTag,
                                     EntityTag,
                                     DimensionsTraits > SubentityTraits;

protected:
   typedef typename SubentityTraits::ContainerType               ContainerType;
   typedef typename SubentityTraits::SharedArrayType             SharedArrayType;

   SharedArrayType getSubentityIndices( DimensionsTraits ) const   { return SharedArrayType( this->subentityVertices); }

   ContainerType&  getSubentityIndices( DimensionsTraits ) { return this->subentityVertices; }

private:
   ContainerType subentityVertices;
};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshSubentityStorageLayer< ConfigTag,
                                    EntityTag,
                                    tnlDimensionsTraits< 0 >,
                                    tnlStorageTraits< false > >
{
};


#endif /* TNLMESHSUBENTITYSTORAGELAYER_H_ */
