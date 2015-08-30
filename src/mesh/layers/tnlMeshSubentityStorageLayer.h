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
#include <mesh/tnlDimensionsTag.h>
#include <mesh/traits/tnlStorageTraits.h>
#include <mesh/traits/tnlMeshSubentitiesTraits.h>
#include <mesh/tnlMeshEntityOrientation.h>

template< typename MeshConfig,
          typename EntityTag,
          typename DimensionsTag,
          typename SubentityStorageTag = 
             typename tnlMeshSubentitiesTraits< MeshConfig,
                                                EntityTag,
                                                DimensionsTag::value >::SubentityStorageTag,
          typename SubentityOrientationStorage =
             tnlStorageTraits< tnlMeshTraits< MeshConfig>::
                template SubentityTraits< EntityTag, DimensionsTag::value >::orientationEnabled > >
class tnlMeshSubentityStorageLayer;


template< typename MeshConfig,
          typename EntityTag >
class tnlMeshSubentityStorageLayers
   : public tnlMeshSubentityStorageLayer< MeshConfig,
                                          EntityTag,
                                          tnlDimensionsTag< EntityTag::dimensions - 1 > >
{
};


template< typename MeshConfig,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshSubentityStorageLayer< MeshConfig,
                                    EntityTag,
                                    DimensionsTag,
                                    tnlStorageTraits< true >,
                                    tnlStorageTraits< true > >
   : public tnlMeshSubentityStorageLayer< MeshConfig,
                                          EntityTag,
                                          typename DimensionsTag::Decrement >
{
   typedef tnlMeshSubentityStorageLayer< MeshConfig,
                                         EntityTag,
                                         typename DimensionsTag::Decrement > BaseType;

   typedef tnlMeshSubentitiesTraits< MeshConfig,
                                     EntityTag,
                                     DimensionsTag::value > SubentityTraits;

   protected:

   typedef typename SubentityTraits::ContainerType        ContainerType;
   typedef typename SubentityTraits::SharedContainerType  SharedContainerType;
   typedef typename ContainerType::ElementType            GlobalIndexType;
   typedef int                                            LocalIndexType;
   typedef typename SubentityTraits::IdArrayType          IdArrayType;
   typedef typename SubentityTraits::OrientationArrayType  OrientationArrayType;
   typedef typename tnlMeshTraits< MeshConfig >::IdPermutationArrayAccessorType   IdPermutationArrayAccessorType;

   tnlMeshSubentityStorageLayer()
   {
      this->subentitiesIndices.setValue( -1 );
      this->sharedSubentitiesIndices.bind( this->subentitiesIndices );
      this->sharedSubentitiesIndices.setName( "sharedSubentitiesIndices" );
      //this->subentitiesIndices.setName( "subentitiesIndices" );
   }

   /*~tnlMeshSubentityStorageLayer()
   {
      cout << "      Destroying " << this->sharedSubentitiesIndices.getSize() << " subentities with "<< DimensionsTag::value << " dimensions." << endl;
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
         cerr << "Saving of the entity subentities layer with " << DimensionsTag::value << " failed." << endl;
         return false;
      }
      return true;
   }

   bool load( tnlFile& file )
   {
      if( ! BaseType::load( file ) ||
          ! this->subentitiesIndices.load( file ) )
      {
         cerr << "Loading of the entity subentities layer with " << DimensionsTag::value << " failed." << endl;
         return false;
      }
      this->sharedSubentitiesIndices.bind( this->subentitiesIndices );
      return true;
   }

   void print( ostream& str ) const
   {
      BaseType::print( str );
      str << endl;
      str << "\t Subentities with " << DimensionsTag::value << " dimensions are: " << subentitiesIndices << ".";
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
   void setSubentityIndex( DimensionsTag,
                           const LocalIndexType localIndex,
                           const GlobalIndexType globalIndex )
   {
      this->subentitiesIndices[ localIndex ] = globalIndex;
   }

   GlobalIndexType getSubentityIndex( DimensionsTag,
                                      const LocalIndexType localIndex ) const
   {
      return this->subentitiesIndices[ localIndex ];
   }

   SharedContainerType& getSubentitiesIndices( DimensionsTag )
   {
      tnlAssert( this->subentitiesIndices.getData() == this->sharedSubentitiesIndices.getData(), );
      return this->sharedSubentitiesIndices;
   }

   const SharedContainerType& getSubentitiesIndices( DimensionsTag ) const
   {
      tnlAssert( this->subentitiesIndices.getData() == this->sharedSubentitiesIndices.getData(), );
      return this->sharedSubentitiesIndices;
   }
   
   using BaseType::subentityIdsArray;
   IdArrayType& subentityIdsArray( DimensionsTag ) { return this->subentitiesIndices; }
   
   using BaseType::subentityOrientation;
   IdPermutationArrayAccessorType subentityOrientation( DimensionsTag, LocalIndexType index) const
   {
      tnlAssert( 0 <= index && index < SubentityTraits::count, );
       
      return this->subentityOrientations[ index ].getSubvertexPermutation();
   }

   using BaseType::subentityOrientationsArray;
	OrientationArrayType& subentityOrientationsArray( DimensionsTag ) { return this->subentityOrientations; }
   
   private:
      IdArrayType subentitiesIndices;

      SharedContainerType sharedSubentitiesIndices;

      OrientationArrayType subentityOrientations;
};


template< typename MeshConfig,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshSubentityStorageLayer< MeshConfig,
                                    EntityTag,
                                    DimensionsTag,
                                    tnlStorageTraits< true >,
                                    tnlStorageTraits< false > >
   : public tnlMeshSubentityStorageLayer< MeshConfig,
                                          EntityTag,
                                          typename DimensionsTag::Decrement >
{
   typedef tnlMeshSubentityStorageLayer< MeshConfig,
                                         EntityTag,
                                         typename DimensionsTag::Decrement > BaseType;

   typedef tnlMeshSubentitiesTraits< MeshConfig,
                                     EntityTag,
                                     DimensionsTag::value > SubentityTraits;

   protected:

   typedef typename SubentityTraits::ContainerType        ContainerType;
   typedef typename SubentityTraits::SharedContainerType  SharedContainerType;
   typedef typename ContainerType::ElementType            GlobalIndexType;
   typedef int                                            LocalIndexType;
   typedef typename SubentityTraits::IdArrayType          IdArrayType;

   tnlMeshSubentityStorageLayer()
   {
      this->subentitiesIndices.setValue( -1 );
      this->sharedSubentitiesIndices.bind( this->subentitiesIndices );
      this->sharedSubentitiesIndices.setName( "sharedSubentitiesIndices" );
      //this->subentitiesIndices.setName( "subentitiesIndices" );
   }

   /*~tnlMeshSubentityStorageLayer()
   {
      cout << "      Destroying " << this->sharedSubentitiesIndices.getSize() << " subentities with "<< DimensionsTag::value << " dimensions." << endl;
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
         cerr << "Saving of the entity subentities layer with " << DimensionsTag::value << " failed." << endl;
         return false;
      }
      return true;
   }

   bool load( tnlFile& file )
   {
      if( ! BaseType::load( file ) ||
          ! this->subentitiesIndices.load( file ) )
      {
         cerr << "Loading of the entity subentities layer with " << DimensionsTag::value << " failed." << endl;
         return false;
      }
      this->sharedSubentitiesIndices.bind( this->subentitiesIndices );
      return true;
   }

   void print( ostream& str ) const
   {
      BaseType::print( str );
      str << endl;
      str << "\t Subentities with " << DimensionsTag::value << " dimensions are: " << subentitiesIndices << ".";
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
   void setSubentityIndex( DimensionsTag,
                           const LocalIndexType localIndex,
                           const GlobalIndexType globalIndex )
   {
      this->subentitiesIndices[ localIndex ] = globalIndex;
   }

   GlobalIndexType getSubentityIndex( DimensionsTag,
                                      const LocalIndexType localIndex ) const
   {
      return this->subentitiesIndices[ localIndex ];
   }

   SharedContainerType& getSubentitiesIndices( DimensionsTag )
   {
      tnlAssert( this->subentitiesIndices.getData() == this->sharedSubentitiesIndices.getData(), );
      return this->sharedSubentitiesIndices;
   }

   const SharedContainerType& getSubentitiesIndices( DimensionsTag ) const
   {
      tnlAssert( this->subentitiesIndices.getData() == this->sharedSubentitiesIndices.getData(), );
      return this->sharedSubentitiesIndices;
   }
   
   using BaseType::subentityIdsArray;
   IdArrayType& subentityIdsArray( DimensionsTag ) { return this->subentitiesIndices; }
   
   using BaseType::subentityOrientationsArray;
   void subentityOrientationsArray() {}
   
   private:
      IdArrayType subentitiesIndices;

      SharedContainerType sharedSubentitiesIndices;

};

template< typename MeshConfig,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshSubentityStorageLayer< MeshConfig,
                                    EntityTag,
                                    DimensionsTag,
                                    tnlStorageTraits< false >,
                                    tnlStorageTraits< false > >
   : public tnlMeshSubentityStorageLayer< MeshConfig,
                                          EntityTag,
                                          typename DimensionsTag::Decrement >
{
};


template< typename MeshConfig,
          typename EntityTag >
class tnlMeshSubentityStorageLayer< MeshConfig,
                                    EntityTag,
                                    tnlDimensionsTag< 0 >,
                                    tnlStorageTraits< true >,
                                    tnlStorageTraits< false > >
{
   typedef tnlDimensionsTag< 0 >                           DimensionsTag;

   typedef tnlMeshSubentitiesTraits< MeshConfig,
                                     EntityTag,
                                     DimensionsTag::value > SubentityTraits;

   protected:

   typedef typename SubentityTraits::ContainerType             ContainerType;
   typedef typename SubentityTraits::SharedContainerType       SharedContainerType;
   typedef typename ContainerType::ElementType                 GlobalIndexType;
   typedef int                                                 LocalIndexType;
   typedef typename SubentityTraits::IdArrayType               IdArrayType;

   tnlMeshSubentityStorageLayer()
   {
      this->verticesIndices.setValue( -1 );
      this->sharedVerticesIndices.bind( this->verticesIndices );
   }

   /*~tnlMeshSubentityStorageLayer()
   {
      cout << "      Destroying " << this->sharedVerticesIndices.getSize() << " subentities with "<< DimensionsTag::value << " dimensions." << endl;
   }*/


   tnlMeshSubentityStorageLayer& operator = ( const tnlMeshSubentityStorageLayer& layer )
   {
      this->verticesIndices = layer.verticesIndices;
      return *this;
   }

   bool save( tnlFile& file ) const
   {
      if( ! this->verticesIndices.save( file ) )
      {
         cerr << "Saving of the entity subentities layer with " << DimensionsTag::value << " failed." << endl;
         return false;
      }
      return true;
   }

   bool load( tnlFile& file )
   {
      if( ! this->verticesIndices.load( file ) )
      {
         cerr << "Loading of the entity subentities layer with " << DimensionsTag::value << " failed." << endl;
         return false;
      }
      this->sharedVerticesIndices.bind( this->verticesIndices );
      return true;
   }

   void print( ostream& str ) const
   {
      str << "\t Subentities with " << DimensionsTag::value << " dimensions are: " << this->verticesIndices << ".";
   }

   bool operator==( const tnlMeshSubentityStorageLayer& layer  ) const
   {
      return ( verticesIndices == layer.verticesIndices );
   }

   GlobalIndexType getSubentityIndex( DimensionsTag,
                                      const LocalIndexType localIndex ) const
   {
      return this->verticesIndices[ localIndex ];
   }
   void setSubentityIndex( DimensionsTag,
                           const LocalIndexType localIndex,
                           const GlobalIndexType globalIndex )
   {
      this->verticesIndices[ localIndex ] = globalIndex;
   }

   SharedContainerType& getSubentitiesIndices( DimensionsTag )
   {
      tnlAssert( this->verticesIndices.getData() == this->sharedVerticesIndices.getData(), );
      return this->sharedVerticesIndices;
   }

   const SharedContainerType& getSubentitiesIndices( DimensionsTag ) const
   {
      tnlAssert( this->verticesIndices.getData() == this->sharedVerticesIndices.getData(), );
      return this->sharedVerticesIndices;
   }

   IdArrayType& subentityIdsArray( DimensionsTag ) { return this->subentitiesIndices; }
   
   protected:
      
      /***
       *  Necessary because of 'using TBase::...;' in the derived classes
       */
	   void subentityOrientation()       {}
	   void subentityOrientationsArray() {}

      IdArrayType verticesIndices;

      SharedContainerType sharedVerticesIndices;
};

template< typename MeshConfig,
          typename EntityTag >
class tnlMeshSubentityStorageLayer< MeshConfig,
                                    EntityTag,
                                    tnlDimensionsTag< 0 >,
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
