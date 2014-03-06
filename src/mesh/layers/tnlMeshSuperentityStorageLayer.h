/***************************************************************************
                          tnlMeshSuperentityStorageLayer.h  -  description
                             -------------------
    begin                : Feb 13, 2014
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

#ifndef TNLMESHSUPERENTITYSTORAGELAYER_H_
#define TNLMESHSUPERENTITYSTORAGELAYER_H_

#include <core/tnlFile.h>
#include <mesh/traits/tnlDimensionsTraits.h>
#include <mesh/traits/tnlStorageTraits.h>
#include <mesh/traits/tnlMeshTraits.h>
#include <mesh/traits/tnlMeshSuperentitiesTraits.h>

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTraits,
          typename SuperentityStorageTag =
             typename tnlMeshSuperentitiesTraits< ConfigTag,
                                                  EntityTag,
                                                  DimensionsTraits >::SuperentityStorageTag >
class tnlMeshSuperentityStorageLayer;

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshSuperentityStorageLayers
   : public tnlMeshSuperentityStorageLayer< ConfigTag,
                                            EntityTag,
                                            typename tnlMeshTraits< ConfigTag >::DimensionsTraits >
{
};

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTraits >
class tnlMeshSuperentityStorageLayer< ConfigTag,
                                      EntityTag,
                                      DimensionsTraits,
                                      tnlStorageTraits< true > >
   : public tnlMeshSuperentityStorageLayer< ConfigTag,
                                            EntityTag,
                                            typename DimensionsTraits::Previous >
{
   typedef
      tnlMeshSuperentityStorageLayer< ConfigTag,
                                      EntityTag,
                                      typename DimensionsTraits::Previous >  BaseType;

   typedef
      tnlMeshSuperentitiesTraits< ConfigTag, EntityTag, DimensionsTraits >          SuperentityTag;

   protected:

   typedef typename SuperentityTag::ContainerType       ContainerType;
   typedef typename SuperentityTag::SharedContainerType SharedContainerType;
   typedef typename ContainerType::ElementType          GlobalIndexType;
   typedef int                                          LocalIndexType;

   bool save( tnlFile& file ) const
   {
      if( ! BaseType::save( file ) ||
          ! this->superentitiesIndices.save( file ) )
         return false;
      return true;
   }

   bool load( tnlFile& file )
   {
      if( ! BaseType::load( file ) ||
          ! this->superentitiesIndices.load( file ) )
         return false;
      return true;
   }

   void print( ostream& str ) const
   {
   }

   /****
     * Make visible setters and getters of the lower superentities
     */
    using BaseType::setNumberOfSuperentities;
    using BaseType::getNumberOfSuperentities;
    using BaseType::getSuperentityIndex;
    using BaseType::setSuperentityIndex;
    using BaseType::getSuperentitiesIndecis;


    /****
     * Define setter/getter for the current level of the superentities
     */
    bool setNumberOfSuperentities( DimensionsTraits,
                                   const LocalIndexType size )
    {
       if( ! this->superentitiesIndices.setSize( size ) )
          return false;
       this->sharedSuperentitiesIndices.bind( this->superentitiesIndices );
       return true;
    }

    LocalIndexType getNumberOfSuperentities( DimensionsTraits ) const
    {
       return this->superentitiesIndices.getSize();
    }

    void setSuperentityIndex( DimensionsTraits,
                              const LocalIndexType localIndex,
                              const GlobalIndexType globalIndex )
    {
       this->superentitiesIndices[ localIndex ] = globalIndex;
    }

    GlobalIndexType getSuperentityIndex( DimensionsTraits,
                                         const LocalIndexType localIndex ) const
    {
       return this->superentitiesIndices[ localIndex ];
    }

    SharedContainerType& getSuperentitiesIndecis()
    {
       return this->sharedSuperentitiesIndecis;
    }

    const SharedContainerType& getSuperentitiesIndecis() const
    {
       return this->sharedSuperentitiesIndecis;
    }

    private:

    ContainerType superentitiesIndices;

    SharedContainerType sharedSuperentitiesIndices;
};

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTraits >
class tnlMeshSuperentityStorageLayer< ConfigTag,
                                      EntityTag,
                                      DimensionsTraits,
                                      tnlStorageTraits< false > >
   : public tnlMeshSuperentityStorageLayer< ConfigTag,
                                            EntityTag,
                                            typename DimensionsTraits::Previous >
{
   public:

};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshSuperentityStorageLayer< ConfigTag,
                                      EntityTag,
                                      tnlDimensionsTraits< EntityTag::dimensions >,
                                      tnlStorageTraits< false > >
{
   typedef tnlDimensionsTraits< EntityTag::dimensions >        DimensionsTraits;

   typedef tnlMeshSuperentitiesTraits< ConfigTag,
                                       EntityTag,
                                       DimensionsTraits >      SuperentityTag;

   protected:

   typedef typename SuperentityTag::ContainerType              ContainerType;
   typedef typename ContainerType::ElementType                 GlobalIndexType;
   typedef int                                                 LocalIndexType;

   /****
    * These methods are due to 'using BaseType::...;' in the derived classes.
    */
   bool setNumberOfSuperentities( DimensionsTraits,
                                   const LocalIndexType size );
   LocalIndexType getNumberOfSuperentities( DimensionsTraits ) const;
   GlobalIndexType getSuperentityIndex( DimensionsTraits,
                                        const LocalIndexType localIndex ){}
   void setSuperentityIndex( DimensionsTraits,
                             const LocalIndexType localIndex,
                             const GlobalIndexType globalIndex ) {}

   void print( ostream& str ) const{}

   ContainerType& getSuperentitiesIndecis(){}

   const ContainerType& getSuperentitiesIndecis() const{}
};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshSuperentityStorageLayer< ConfigTag,
                                      EntityTag,
                                      tnlDimensionsTraits< EntityTag::dimensions >,
                                      tnlStorageTraits< true > >
{
   typedef tnlDimensionsTraits< EntityTag::dimensions >        DimensionsTraits;

   typedef tnlMeshSuperentitiesTraits< ConfigTag,
                                       EntityTag,
                                       DimensionsTraits >      SuperentityTag;

   protected:

   typedef typename SuperentityTag::ContainerType              ContainerType;
   typedef typename ContainerType::ElementType                 GlobalIndexType;
   typedef int                                                 LocalIndexType;

   /****
    * These methods are due to 'using BaseType::...;' in the derived classes.
    */
   bool setNumberOfSuperentities( DimensionsTraits,
                                   const LocalIndexType size );
   LocalIndexType getNumberOfSuperentities( DimensionsTraits ) const;
   GlobalIndexType getSuperentityIndex( DimensionsTraits,
                                        const LocalIndexType localIndex ){}
   void setSuperentityIndex( DimensionsTraits,
                             const LocalIndexType localIndex,
                             const GlobalIndexType globalIndex ) {}

   void print( ostream& str ) const{}

   ContainerType& getSuperentitiesIndecis(){}

   const ContainerType& getSuperentitiesIndecis() const{}


};

/*template< typename ConfigTag,
          typename EntityTag >
class tnlMeshSuperentityStorageLayer< ConfigTag,
                                      EntityTag,
                                      tnlDimensionsTraits< 0 >,
                                      tnlStorageTraits< false > >
{
};*/

#endif /* TNLMESHSUPERENTITYSTORAGELAYER_H_ */
