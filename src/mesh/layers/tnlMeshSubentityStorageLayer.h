/***************************************************************************
                          tnlMeshSubentityStorageLayer.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLMESHSUBENTITYSTORAGELAYER_H_
#define TNLMESHSUBENTITYSTORAGELAYER_H_

#include <core/tnlFile.h>
#include <mesh/tnlDimensionsTag.h>
#include <mesh/traits/tnlMeshSubentityTraits.h>
#include <mesh/tnlMeshEntityOrientation.h>

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag,
          bool SubentityStorage =
            tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag::value >::storageEnabled,
          bool SubentityOrientationStorage =
            tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag::value >::orientationEnabled >
class tnlMeshSubentityStorageLayer;


template< typename MeshConfig,
          typename EntityTopology >
class tnlMeshSubentityStorageLayers
   : public tnlMeshSubentityStorageLayer< MeshConfig,
                                          EntityTopology,
                                          tnlDimensionsTag< EntityTopology::dimensions - 1 > >
{
};


template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag >
class tnlMeshSubentityStorageLayer< MeshConfig,
                                    EntityTopology,
                                    DimensionsTag,
                                    true,
                                    true >
   : public tnlMeshSubentityStorageLayer< MeshConfig,
                                          EntityTopology,
                                          typename DimensionsTag::Decrement >
{
   typedef tnlMeshSubentityStorageLayer< MeshConfig,
                                         EntityTopology,
                                         typename DimensionsTag::Decrement > BaseType;

   protected:

   static const int Dimensions = DimensionsTag::value;
   typedef tnlMeshTraits< MeshConfig >                                                   MeshTraits;
   typedef typename MeshTraits::template SubentityTraits< EntityTopology, Dimensions >   SubentityTraits;
   typedef typename MeshTraits::GlobalIndexType                                          GlobalIndexType;
   typedef typename MeshTraits::LocalIndexType                                           LocalIndexType;
   typedef typename SubentityTraits::IdArrayType                                         IdArrayType;
   typedef typename SubentityTraits::OrientationArrayType                                OrientationArrayType;
   typedef typename MeshTraits::IdPermutationArrayAccessorType                           IdPermutationArrayAccessorType;

   tnlMeshSubentityStorageLayer()
   {
      this->subentitiesIndices.setValue( -1 );
   }

   ~tnlMeshSubentityStorageLayer()
   {
      //cout << "      Destroying " << this->sharedSubentitiesIndices.getSize() << " subentities with "<< DimensionsTag::value << " dimensions." << endl;
   }

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

      OrientationArrayType subentityOrientations;
};


template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag >
class tnlMeshSubentityStorageLayer< MeshConfig,
                                    EntityTopology,
                                    DimensionsTag,
                                    true,
                                    false >
   : public tnlMeshSubentityStorageLayer< MeshConfig,
                                          EntityTopology,
                                          typename DimensionsTag::Decrement >
{
   typedef tnlMeshSubentityStorageLayer< MeshConfig,
                                         EntityTopology,
                                         typename DimensionsTag::Decrement > BaseType;

   protected:
 
   static const int Dimensions = DimensionsTag::value;
   typedef tnlMeshTraits< MeshConfig >                                                   MeshTraits;
   typedef typename MeshTraits::template SubentityTraits< EntityTopology, Dimensions >   SubentityTraits;
   typedef typename MeshTraits::GlobalIndexType                                          GlobalIndexType;
   typedef typename MeshTraits::LocalIndexType                                           LocalIndexType;
   typedef typename SubentityTraits::IdArrayType                                         IdArrayType;
   typedef typename SubentityTraits::OrientationArrayType                                OrientationArrayType;
   typedef typename MeshTraits::IdPermutationArrayAccessorType                           IdPermutationArrayAccessorType;

   tnlMeshSubentityStorageLayer()
   {
      this->subentitiesIndices.setValue( -1 );
   }

   ~tnlMeshSubentityStorageLayer()
   {
      //cout << "      Destroying " << this->sharedSubentitiesIndices.getSize() << " subentities with "<< DimensionsTag::value << " dimensions." << endl;
   }

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

   using BaseType::subentityIdsArray;
   IdArrayType& subentityIdsArray( DimensionsTag ) { return this->subentitiesIndices; }
 
   using BaseType::subentityOrientationsArray;
   void subentityOrientationsArray() {}
 
   private:
      IdArrayType subentitiesIndices;
};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag >
class tnlMeshSubentityStorageLayer< MeshConfig,
                                    EntityTopology,
                                    DimensionsTag,
                                    false,
                                    false >
   : public tnlMeshSubentityStorageLayer< MeshConfig,
                                          EntityTopology,
                                          typename DimensionsTag::Decrement >
{
};


template< typename MeshConfig,
          typename EntityTopology >
class tnlMeshSubentityStorageLayer< MeshConfig,
                                    EntityTopology,
                                    tnlDimensionsTag< 0 >,
                                    true,
                                    false >
{
   typedef tnlDimensionsTag< 0 >                           DimensionsTag;

   protected:
   static const int Dimensions = 0;
   typedef tnlMeshTraits< MeshConfig >                                                   MeshTraits;
   typedef typename MeshTraits::template SubentityTraits< EntityTopology, Dimensions >   SubentityTraits;
   typedef typename MeshTraits::GlobalIndexType                                          GlobalIndexType;
   typedef typename MeshTraits::LocalIndexType                                           LocalIndexType;
   typedef typename SubentityTraits::IdArrayType                                         IdArrayType;

   tnlMeshSubentityStorageLayer()
   {
      this->verticesIndices.setValue( -1 );
   }

   ~tnlMeshSubentityStorageLayer()
   {
      //cout << "      Destroying " << this->sharedVerticesIndices.getSize() << " subentities with "<< DimensionsTag::value << " dimensions." << endl;
   }


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

   IdArrayType& subentityIdsArray( DimensionsTag ) { return this->verticesIndices; }
 
   protected:
 
      /***
       *  Necessary because of 'using TBase::...;' in the derived classes
       */
	   void subentityOrientation()       {}
	   void subentityOrientationsArray() {}

      IdArrayType verticesIndices;
};

template< typename MeshConfig,
          typename EntityTopology >
class tnlMeshSubentityStorageLayer< MeshConfig,
                                    EntityTopology,
                                    tnlDimensionsTag< 0 >,
                                    false,
                                    false >
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
