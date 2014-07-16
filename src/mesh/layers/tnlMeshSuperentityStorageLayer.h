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

   /****
     * Make visible setters and getters of the lower superentities
     */
    using BaseType::setNumberOfSuperentities;
    using BaseType::getNumberOfSuperentities;
    using BaseType::getSuperentityIndex;
    using BaseType::setSuperentityIndex;
    using BaseType::getSuperentitiesIndices;

    tnlMeshSuperentityStorageLayer()
    {
       this->superentitiesIndices.setName( tnlString( "tnlMeshSuperentityStorageLayer < " ) + tnlString( DimensionsTraits::value ) + " >::superentitiesIndices" );
       this->sharedSuperentitiesIndices.setName( tnlString( "tnlMeshSuperentityStorageLayer < " ) + tnlString( DimensionsTraits::value ) + " >::sharedSuperentitiesIndices" );
    }

    /*~tnlMeshSuperentityStorageLayer()
    {
       cerr << "      Destroying " << this->superentitiesIndices.getSize() << " superentities with "<< DimensionsTraits::value << " dimensions." << endl;
       cerr << "         this->superentitiesIndices.getName() = " << this->superentitiesIndices.getName() << endl;
       cerr << "         this->sharedSuperentitiesIndices.getName() = " << this->sharedSuperentitiesIndices.getName() << endl;
    }*/

    tnlMeshSuperentityStorageLayer& operator = ( const tnlMeshSuperentityStorageLayer& layer )
    {
       this->superentitiesIndices.setSize( layer.superentitiesIndices.getSize() );
       this->superentitiesIndices = layer.superentitiesIndices;
       this->sharedSuperentitiesIndices.bind( this->superentitiesIndices );
       return *this;
    }

    /****
     * Define setter/getter for the current level of the superentities
     */
    bool setNumberOfSuperentities( DimensionsTraits,
                                   const LocalIndexType size )
    {
       if( ! this->superentitiesIndices.setSize( size ) )
          return false;
       this->superentitiesIndices.setValue( -1 );
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

    SharedContainerType& getSuperentitiesIndices( DimensionsTraits )
    {
       return this->sharedSuperentitiesIndices;
    }

    const SharedContainerType& getSuperentitiesIndices( DimensionsTraits ) const
    {
       return this->sharedSuperentitiesIndices;
    }

    bool save( tnlFile& file ) const
    {
       if( ! BaseType::save( file ) ||
           ! this->superentitiesIndices.save( file ) )
       {
          //cerr << "Saving of the entity superentities layer with " << DimensionsTraits::value << " failed." << endl;
          return false;
       }
       return true;
    }

    bool load( tnlFile& file )
    {
       if( ! BaseType::load( file ) ||
           ! this->superentitiesIndices.load( file ) )
       {
          //cerr << "Loading of the entity superentities layer with " << DimensionsTraits::value << " failed." << endl;
          return false;
       }
       return true;
    }

    void print( ostream& str ) const
    {
       BaseType::print( str );
       str << endl << "\t Superentities with " << DimensionsTraits::value << " dimensions are: " << this->superentitiesIndices << ".";
    }

    bool operator==( const tnlMeshSuperentityStorageLayer& layer  ) const
    {
       return ( BaseType::operator==( layer ) &&
                superentitiesIndices == layer.superentitiesIndices );
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

   typedef tnlMeshSuperentityStorageLayer< ConfigTag,
                                           EntityTag,
                                           DimensionsTraits,
                                           tnlStorageTraits< false > > ThisType;

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

   bool operator==( const ThisType& layer  ) const
   {
      return true;
   }

   ContainerType& getSuperentitiesIndices(){}

   const ContainerType& getSuperentitiesIndices() const{}

   bool save( tnlFile& file ) const
   {
      return true;
   }

   bool load( tnlFile& file )
   {
      return true;
   }

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
   typedef tnlMeshSuperentityStorageLayer< ConfigTag,
                                           EntityTag,
                                           DimensionsTraits,
                                           tnlStorageTraits< true > > ThisType;

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

   bool operator==( const ThisType& layer  ) const
   {
      return true;
   }

   ContainerType& getSuperentitiesIndices(){}

   const ContainerType& getSuperentitiesIndices() const{}

   bool save( tnlFile& file ) const
   {
      return true;
   }

   bool load( tnlFile& file )
   {
      return true;
   }
};

#endif /* TNLMESHSUPERENTITYSTORAGELAYER_H_ */
