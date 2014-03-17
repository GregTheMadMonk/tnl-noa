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
      this->sharedSubentitiesIndices.bind( this->subentitiesIndices );
      this->sharedSubentitiesIndices.setName( "sharedSubentitiesIndices" );
      //this->subentitiesIndices.setName( "subentitiesIndices" );
   }

   /*~tnlMeshSubentityStorageLayer()
   {
      cout << "      Destroying " << this->sharedSubentitiesIndices.getSize() << " subentities with "<< DimensionsTraits::value << " dimensions." << endl;
   }*/

   tnlMeshSubentityStorageLayer& operator = ( const tnlMeshSubentityStorageLayer& layer )
   {
      BaseType::operator=( layer );
      this->subentitiesIndices = layer.subentitiesIndices;
      return *this;
   }

   bool save( tnlFile& file ) const
   {
      if( ! BaseType::save( file ) ||
          ! this->subentitiesIndices.save( file ) )
      {
         cerr << "Saving of the entity subentities layer with " << DimensionsTraits::value << " failed." << endl;
         return false;
      }
      return true;
   }

   bool load( tnlFile& file )
   {
      if( ! BaseType::load( file ) ||
          ! this->subentitiesIndices.load( file ) )
      {
         cerr << "Loading of the entity subentities layer with " << DimensionsTraits::value << " failed." << endl;
         return false;
      }
      this->sharedSubentitiesIndices.bind( this->subentitiesIndices );
      return true;
   }

   void print( ostream& str ) const
   {
      BaseType::print( str );
      str << endl;
      str << "\t Subentities with " << DimensionsTraits::value << " dimensions are: " << subentitiesIndices << ".";
   }

   bool operator==( const tnlMeshSubentityStorageLayer& layer  ) const
   {
      return ( BaseType::operator==( layer ) &&
               subentitiesIndices == layer.subentitiesIndices );
   }

   /****
    * Make visible setters and getters of the lower subentities
    */
   using BaseType::getSubentityIndex;
   using BaseType::setSubentityIndex;
   using BaseType::getSubentitiesIndices;

   /****
    * Define setter/getter for the current level of the subentities
    */
   void setSubentityIndex( DimensionsTraits,
                           const LocalIndexType localIndex,
                           const GlobalIndexType globalIndex )
   {
      this->subentitiesIndices[ localIndex ] = globalIndex;
   }

   GlobalIndexType getSubentityIndex( DimensionsTraits,
                                      const LocalIndexType localIndex ) const
   {
      return this->subentitiesIndices[ localIndex ];
   }

   SharedContainerType& getSubentitiesIndices( DimensionsTraits )
   {
      tnlAssert( this->subentitiesIndices.getData() == this->sharedSubentitiesIndices.getData(), );
      return this->sharedSubentitiesIndices;
   }

   const SharedContainerType& getSubentitiesIndices( DimensionsTraits ) const
   {
      tnlAssert( this->subentitiesIndices.getData() == this->sharedSubentitiesIndices.getData(), );
      return this->sharedSubentitiesIndices;
   }

   private:
   ContainerType subentitiesIndices;

   SharedContainerType sharedSubentitiesIndices;

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
      this->sharedVerticesIndices.bind( this->verticesIndices );
   }

   /*~tnlMeshSubentityStorageLayer()
   {
      cout << "      Destroying " << this->sharedVerticesIndices.getSize() << " subentities with "<< DimensionsTraits::value << " dimensions." << endl;
   }*/


   tnlMeshSubentityStorageLayer& operator = ( const tnlMeshSubentityStorageLayer& layer )
   {
      this->verticesIndices = layer.verticesIndices;
      cout << " layer.verticesIndices = " << layer.verticesIndices << endl;
      cout << " this->verticesIndices = " << this->verticesIndices << endl;
      return *this;
   }

   bool save( tnlFile& file ) const
   {
      if( ! this->verticesIndices.save( file ) )
      {
         cerr << "Saving of the entity subentities layer with " << DimensionsTraits::value << " failed." << endl;
         return false;
      }
      return true;
   }

   bool load( tnlFile& file )
   {
      if( ! this->verticesIndices.load( file ) )
      {
         cerr << "Loading of the entity subentities layer with " << DimensionsTraits::value << " failed." << endl;
         return false;
      }
      this->sharedVerticesIndices.bind( this->verticesIndices );
      return true;
   }

   void print( ostream& str ) const
   {
      str << "\t Subentities with " << DimensionsTraits::value << " dimensions are: " << this->verticesIndices << ".";
   }

   bool operator==( const tnlMeshSubentityStorageLayer& layer  ) const
   {
      return ( verticesIndices == layer.verticesIndices );
   }

   GlobalIndexType getSubentityIndex( DimensionsTraits,
                                      const LocalIndexType localIndex ) const
   {
      return this->verticesIndices[ localIndex ];
   }
   void setSubentityIndex( DimensionsTraits,
                           const LocalIndexType localIndex,
                           const GlobalIndexType globalIndex )
   {
      this->verticesIndices[ localIndex ] = globalIndex;
   }

   SharedContainerType& getSubentitiesIndices( DimensionsTraits )
   {
      tnlAssert( this->verticesIndices.getData() == this->sharedVerticesIndices.getData(), );
      return this->sharedVerticesIndices;
   }

   const SharedContainerType& getSubentitiesIndices( DimensionsTraits ) const
   {
      tnlAssert( this->verticesIndices.getData() == this->sharedVerticesIndices.getData(), );
      return this->sharedVerticesIndices;
   }

   private:

   ContainerType verticesIndices;

   SharedContainerType sharedVerticesIndices;
};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshSubentityStorageLayer< ConfigTag,
                                    EntityTag,
                                    tnlDimensionsTraits< 0 >,
                                    tnlStorageTraits< false > >
{
   public:

   bool save( tnlFile& file ) const
   {
      return true;
   }

   bool load( tnlFile& file )
   {
      return true;
   }

};


#endif /* TNLMESHSUBENTITYSTORAGELAYER_H_ */
