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
#include <mesh/tnlDimensionsTag.h>
#include <mesh/traits/tnlStorageTraits.h>
#include <mesh/traits/tnlMeshTraits.h>
#include <mesh/traits/tnlMeshConfigTraits.h>
#include <mesh/traits/tnlMeshSuperentitiesTraits.h>

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag,
          typename SuperentityStorageTag =
             typename tnlMeshSuperentitiesTraits< ConfigTag,
                                                  EntityTag,
                                                  DimensionsTag >::SuperentityStorageTag >
class tnlMeshSuperentityStorageLayer;

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshSuperentityStorageLayers
   : public tnlMeshSuperentityStorageLayer< ConfigTag,
                                            EntityTag,
                                            typename tnlMeshTraits< ConfigTag >::DimensionsTag >
{
};

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshSuperentityStorageLayer< ConfigTag,
                                      EntityTag,
                                      DimensionsTag,
                                      tnlStorageTraits< true > >
   : public tnlMeshSuperentityStorageLayer< ConfigTag,
                                            EntityTag,
                                            typename DimensionsTag::Decrement >
{
   typedef
      tnlMeshSuperentityStorageLayer< ConfigTag,
                                      EntityTag,
                                      typename DimensionsTag::Decrement >  BaseType;

   typedef
      tnlMeshSuperentitiesTraits< ConfigTag, EntityTag, DimensionsTag >          SuperentityTag;

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
       this->superentitiesIndices.setName( tnlString( "tnlMeshSuperentityStorageLayer < " ) + tnlString( DimensionsTag::value ) + " >::superentitiesIndices" );
       this->sharedSuperentitiesIndices.setName( tnlString( "tnlMeshSuperentityStorageLayer < " ) + tnlString( DimensionsTag::value ) + " >::sharedSuperentitiesIndices" );
    }

    /*~tnlMeshSuperentityStorageLayer()
    {
       cerr << "      Destroying " << this->superentitiesIndices.getSize() << " superentities with "<< DimensionsTag::value << " dimensions." << endl;
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
    bool setNumberOfSuperentities( DimensionsTag,
                                   const LocalIndexType size )
    {
       if( ! this->superentitiesIndices.setSize( size ) )
          return false;
       this->superentitiesIndices.setValue( -1 );
       this->sharedSuperentitiesIndices.bind( this->superentitiesIndices );
       return true;
    }

    LocalIndexType getNumberOfSuperentities( DimensionsTag ) const
    {
       return this->superentitiesIndices.getSize();
    }

    void setSuperentityIndex( DimensionsTag,
                              const LocalIndexType localIndex,
                              const GlobalIndexType globalIndex )
    {
       this->superentitiesIndices[ localIndex ] = globalIndex;
    }

    GlobalIndexType getSuperentityIndex( DimensionsTag,
                                         const LocalIndexType localIndex ) const
    {
       return this->superentitiesIndices[ localIndex ];
    }

    SharedContainerType& getSuperentitiesIndices( DimensionsTag )
    {
       return this->sharedSuperentitiesIndices;
    }

    const SharedContainerType& getSuperentitiesIndices( DimensionsTag ) const
    {
       return this->sharedSuperentitiesIndices;
    }

    bool save( tnlFile& file ) const
    {
       if( ! BaseType::save( file ) ||
           ! this->superentitiesIndices.save( file ) )
       {
          //cerr << "Saving of the entity superentities layer with " << DimensionsTag::value << " failed." << endl;
          return false;
       }
       return true;
    }

    bool load( tnlFile& file )
    {
       if( ! BaseType::load( file ) ||
           ! this->superentitiesIndices.load( file ) )
       {
          //cerr << "Loading of the entity superentities layer with " << DimensionsTag::value << " failed." << endl;
          return false;
       }
       return true;
    }

    void print( ostream& str ) const
    {
       BaseType::print( str );
       str << endl << "\t Superentities with " << DimensionsTag::value << " dimensions are: " << this->superentitiesIndices << ".";
    }

    bool operator==( const tnlMeshSuperentityStorageLayer& layer  ) const
    {
       return ( BaseType::operator==( layer ) &&
                superentitiesIndices == layer.superentitiesIndices );
    }

    private:              

    ContainerType superentitiesIndices;

    SharedContainerType sharedSuperentitiesIndices;
    
   // TODO: this is only for the mesh initializer - fix it
   public:
              
      using BaseType::superentityIdsArray;               
      typename tnlMeshConfigTraits< ConfigTag >::GlobalIdArrayType& superentityIdsArray( DimensionsTag )
      {
         return this->superentitiesIndices;
      }
};

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshSuperentityStorageLayer< ConfigTag,
                                      EntityTag,
                                      DimensionsTag,
                                      tnlStorageTraits< false > >
   : public tnlMeshSuperentityStorageLayer< ConfigTag,
                                            EntityTag,
                                            typename DimensionsTag::Decrement >
{
   public:

};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshSuperentityStorageLayer< ConfigTag,
                                      EntityTag,
                                      tnlDimensionsTag< EntityTag::dimensions >,
                                      tnlStorageTraits< false > >
{
   typedef tnlDimensionsTag< EntityTag::dimensions >        DimensionsTag;

   typedef tnlMeshSuperentitiesTraits< ConfigTag,
                                       EntityTag,
                                       DimensionsTag >      SuperentityTag;

   typedef tnlMeshSuperentityStorageLayer< ConfigTag,
                                           EntityTag,
                                           DimensionsTag,
                                           tnlStorageTraits< false > > ThisType;

   protected:

   typedef typename SuperentityTag::ContainerType              ContainerType;
   typedef typename ContainerType::ElementType                 GlobalIndexType;
   typedef int                                                 LocalIndexType;

   /****
    * These methods are due to 'using BaseType::...;' in the derived classes.
    */
   bool setNumberOfSuperentities( DimensionsTag,
                                   const LocalIndexType size );
   LocalIndexType getNumberOfSuperentities( DimensionsTag ) const;
   GlobalIndexType getSuperentityIndex( DimensionsTag,
                                        const LocalIndexType localIndex ){}
   void setSuperentityIndex( DimensionsTag,
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
   
   template< typename SuperDimensionsTag >
   typename tnlMeshConfigTraits< ConfigTag >::GlobalIdArrayType& superentityIdsArray( DimensionsTag )
   {
      tnlAssert( false, );
      //return this->superentitiesIndices;
   }
};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshSuperentityStorageLayer< ConfigTag,
                                      EntityTag,
                                      tnlDimensionsTag< EntityTag::dimensions >,
                                      tnlStorageTraits< true > >
{
   typedef tnlDimensionsTag< EntityTag::dimensions >        DimensionsTag;

   typedef tnlMeshSuperentitiesTraits< ConfigTag,
                                       EntityTag,
                                       DimensionsTag >      SuperentityTag;
   typedef tnlMeshSuperentityStorageLayer< ConfigTag,
                                           EntityTag,
                                           DimensionsTag,
                                           tnlStorageTraits< true > > ThisType;

   protected:

   typedef typename SuperentityTag::ContainerType              ContainerType;
   typedef typename ContainerType::ElementType                 GlobalIndexType;
   typedef int                                                 LocalIndexType;

   /****
    * These methods are due to 'using BaseType::...;' in the derived classes.
    */
   bool setNumberOfSuperentities( DimensionsTag,
                                   const LocalIndexType size );
   LocalIndexType getNumberOfSuperentities( DimensionsTag ) const;
   GlobalIndexType getSuperentityIndex( DimensionsTag,
                                        const LocalIndexType localIndex ){}
   void setSuperentityIndex( DimensionsTag,
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
   
   template< typename SuperDimensionsTag >
   typename tnlMeshConfigTraits< ConfigTag >::GlobalIdArrayType& superentityIdsArray( DimensionsTag )
   {
      tnlAssert( false, );
      //return this->superentitiesIndices;
   }

};

#endif /* TNLMESHSUPERENTITYSTORAGELAYER_H_ */
