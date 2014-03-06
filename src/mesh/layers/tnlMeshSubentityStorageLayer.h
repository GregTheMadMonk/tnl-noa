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

#include <core/tnlFile.h>
#include <mesh/traits/tnlDimensionsTraits.h>
#include <mesh/traits/tnlStorageTraits.h>
#include <mesh/traits/tnlMeshSubentitiesTraits.h>

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionTraits,
          typename SubentityStorageTag =
                   typename tnlMeshSubentitiesTraits< ConfigTag,
                                                      EntityTag,
                                                      DimensionTraits >::SubentityStorageTag >
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

   typedef typename SubentityTraits::ContainerType        ContainerType;
   typedef typename SubentityTraits::SharedContainerType  SharedContainerType;
   typedef typename ContainerType::ElementType            GlobalIndexType;
   typedef int                                            LocalIndexType;

   tnlMeshSubentityStorageLayer()
   {
      this->sharedSubentitiesIndecis.bind( this->subentitiesIndecis );
   }

   bool save( tnlFile& file ) const
   {
      if( ! BaseType::save( file ) ||
          ! this->subentitiesIndecis.save( file ) )
         return false;
      return true;
   }

   bool load( tnlFile& file )
   {
      if( ! BaseType::load( file ) ||
          ! this->subentitiesIndecis.load( file ) )
         return false;
      this->sharedSubentitiesIndecis.bind( this->subentitiesIndecis );
      return true;
   }

   void print( ostream& str ) const
   {
      BaseType::print( str );
      str << endl;
      str << "\t Subentities with " << DimensionsTraits::value << " dimensions are: " << subentitiesIndecis << ".";
   }

   /****
    * Make visible setters and getters of the lower subentities
    */
   using BaseType::getSubentityIndex;
   using BaseType::setSubentityIndex;
   using BaseType::getSubentitiesIndecis;

   /****
    * Define setter/getter for the current level of the subentities
    */
   void setSubentityIndex( DimensionsTraits,
                           const LocalIndexType localIndex,
                           const GlobalIndexType globalIndex )
   {
      this->subentitiesIndecis[ localIndex ] = globalIndex;
   }

   GlobalIndexType getSubentityIndex( DimensionsTraits,
                                      const LocalIndexType localIndex ) const
   {
      return this->subentitiesIndecis[ localIndex ];
   }

   SharedContainerType& getSubentitiesIndecis( DimensionsTraits )
   {
      return this->sharedSubentitiesIndecis;
   }

   const SharedContainerType& getSubentitiesIndecis( DimensionsTraits ) const
   {
      return this->sharedSubentitiesIndecis;
   }

   private:
   ContainerType subentitiesIndecis;

   SharedContainerType sharedSubentitiesIndecis;

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

   typedef typename SubentityTraits::ContainerType             ContainerType;
   typedef typename SubentityTraits::SharedContainerType       SharedContainerType;
   typedef typename ContainerType::ElementType                 GlobalIndexType;
   typedef int                                                 LocalIndexType;

   tnlMeshSubentityStorageLayer()
   {
      this->sharedVerticesIndecis.bind( this->verticesIndecis );
   }

   bool save( tnlFile& file ) const
   {
      if( ! this->subentitiesVertices.save( file ) )
         return false;
      return true;
   }

   bool load( tnlFile& file )
   {
      if( ! this->subentitiesVertices.load( file ) )
         return false;
      this->sharedVerticesIndecis.bind( this->verticesIndecis );
      return true;
   }

   void print( ostream& str ) const
   {
      str << "\t Subentities with " << DimensionsTraits::value << " dimensions are: " << this->verticesIndecis << ".";
   }

   GlobalIndexType getSubentityIndex( DimensionsTraits,
                                      const LocalIndexType localIndex ) const
   {
      return this->verticesIndecis[ localIndex ];
   }
   void setSubentityIndex( DimensionsTraits,
                           const LocalIndexType localIndex,
                           const GlobalIndexType globalIndex )
   {
      this->verticesIndecis[ localIndex ] = globalIndex;
   }

   SharedContainerType& getSubentitiesIndecis( DimensionsTraits )
   {
      return this->sharedVerticesIndecis;
   }

   const SharedContainerType& getSubentitiesIndecis( DimensionsTraits ) const
   {
      return this->sharedVerticesIndecis;
   }

   private:

   ContainerType verticesIndecis;

   SharedContainerType sharedVerticesIndecis;
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
