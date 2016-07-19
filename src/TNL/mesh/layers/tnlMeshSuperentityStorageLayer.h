/***************************************************************************
                          tnlMeshSuperentityStorageLayer.h  -  description
                             -------------------
    begin                : Feb 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/core/tnlFile.h>
#include <TNL/mesh/tnlDimensionsTag.h>
#include <TNL/mesh/traits/tnlMeshTraits.h>
#include <TNL/mesh/traits/tnlMeshSuperentityTraits.h>

namespace TNL {

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag,
          bool SuperentityStorage =
             tnlMeshSuperentityTraits< MeshConfig, EntityTopology, DimensionsTag::value >::storageEnabled >
class tnlMeshSuperentityStorageLayer;

template< typename MeshConfig,
          typename EntityTopology >
class tnlMeshSuperentityStorageLayers
   : public tnlMeshSuperentityStorageLayer< MeshConfig,
                                            EntityTopology,
                                            tnlDimensionsTag< tnlMeshTraits< MeshConfig >::meshDimensions > >
{
};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag >
class tnlMeshSuperentityStorageLayer< MeshConfig, EntityTopology, DimensionsTag, true >
   : public tnlMeshSuperentityStorageLayer< MeshConfig, EntityTopology, typename DimensionsTag::Decrement >
{
   typedef
      tnlMeshSuperentityStorageLayer< MeshConfig, EntityTopology, typename DimensionsTag::Decrement >  BaseType;

   static const int Dimensions = DimensionsTag::value;
   typedef tnlMeshTraits< MeshConfig >                                                   MeshTraits;
   typedef typename MeshTraits::template SuperentityTraits< EntityTopology, Dimensions > SuperentityTraits;

   protected:

   typedef typename SuperentityTraits::StorageArrayType       StorageArrayType;
   typedef typename SuperentityTraits::AccessArrayType        AccessArrayType;
   typedef typename SuperentityTraits::GlobalIndexType        GlobalIndexType;
   typedef typename SuperentityTraits::LocalIndexType         LocalIndexType;

   typedef typename SuperentityTraits::StorageNetworkType   StorageNetworkType;
 
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
    }

    /*~tnlMeshSuperentityStorageLayer()
    {
       std::cerr << "      Destroying " << this->superentitiesIndices.getSize() << " superentities with "<< DimensionsTag::value << " dimensions." << std::endl;
       std::cerr << "         this->superentitiesIndices.getName() = " << this->superentitiesIndices.getName() << std::endl;
       std::cerr << "         this->sharedSuperentitiesIndices.getName() = " << this->sharedSuperentitiesIndices.getName() << std::endl;
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

    AccessArrayType& getSuperentitiesIndices( DimensionsTag )
    {
       return this->sharedSuperentitiesIndices;
    }

    const AccessArrayType& getSuperentitiesIndices( DimensionsTag ) const
    {
       return this->sharedSuperentitiesIndices;
    }

    bool save( tnlFile& file ) const
    {
       if( ! BaseType::save( file ) ||
           ! this->superentitiesIndices.save( file ) )
       {
          //cerr << "Saving of the entity superentities layer with " << DimensionsTag::value << " failed." << std::endl;
          return false;
       }
       return true;
    }

    bool load( tnlFile& file )
    {
       if( ! BaseType::load( file ) ||
           ! this->superentitiesIndices.load( file ) )
       {
          //cerr << "Loading of the entity superentities layer with " << DimensionsTag::value << " failed." << std::endl;
          return false;
       }
       return true;
    }

    void print( std::ostream& str ) const
    {
       BaseType::print( str );
       str << std::endl << "\t Superentities with " << DimensionsTag::value << " dimensions are: " << this->superentitiesIndices << ".";
    }

    bool operator==( const tnlMeshSuperentityStorageLayer& layer  ) const
    {
       return ( BaseType::operator==( layer ) &&
                superentitiesIndices == layer.superentitiesIndices );
    }

    private:

    StorageArrayType superentitiesIndices;

    AccessArrayType sharedSuperentitiesIndices;
 
    StorageNetworkType storageNetwork;
 
   // TODO: this is only for the mesh initializer - fix it
   public:
 
      using BaseType::superentityIdsArray;
      typename tnlMeshTraits< MeshConfig >::GlobalIdArrayType& superentityIdsArray( DimensionsTag )
      {
         return this->superentitiesIndices;
      }
 
      using BaseType::getStorageNetwork;
      StorageNetworkType& getStorageNetwork( DimensionsTag )
      {
         return this->storageNetwork;
      }
};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag >
class tnlMeshSuperentityStorageLayer< MeshConfig, EntityTopology, DimensionsTag, false >
   : public tnlMeshSuperentityStorageLayer< MeshConfig, EntityTopology, typename DimensionsTag::Decrement >
{
   public:

};

template< typename MeshConfig,
          typename EntityTopology >
class tnlMeshSuperentityStorageLayer< MeshConfig, EntityTopology, tnlDimensionsTag< EntityTopology::dimensions >, false >
{
   static const int Dimensions = EntityTopology::dimensions;
   typedef tnlDimensionsTag< EntityTopology::dimensions >        DimensionsTag;

   typedef tnlMeshSuperentityTraits< MeshConfig, EntityTopology, Dimensions >      SuperentityTraits;

   typedef tnlMeshSuperentityStorageLayer< MeshConfig,
                                           EntityTopology,
                                           DimensionsTag,
                                           false > ThisType;

   protected:

   typedef typename SuperentityTraits::ContainerType              ContainerType;
   typedef typename ContainerType::ElementType                 GlobalIndexType;
   typedef int                                                 LocalIndexType;

   typedef typename SuperentityTraits::StorageNetworkType   StorageNetworkType;
 
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

   void print( std::ostream& str ) const{}

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
 
   typename tnlMeshTraits< MeshConfig >::GlobalIdArrayType& superentityIdsArray( DimensionsTag )
   {
      tnlAssert( false, );
      //return this->superentitiesIndices;
   }

   StorageNetworkType& getStorageNetwork( DimensionsTag )
   {
      tnlAssert( false, );
     //return this->storageNetwork;
   }

};

template< typename MeshConfig,
          typename EntityTopology >
class tnlMeshSuperentityStorageLayer< MeshConfig,
                                      EntityTopology,
                                      tnlDimensionsTag< EntityTopology::dimensions >,
                                      true >
{
   static const int Dimensions = EntityTopology::dimensions;
   typedef tnlDimensionsTag< Dimensions >                          DimensionsTag;

   typedef tnlMeshSuperentityTraits< MeshConfig,
                                     EntityTopology,
                                     Dimensions >               SuperentityTraits;
   typedef tnlMeshSuperentityStorageLayer< MeshConfig,
                                           EntityTopology,
                                           DimensionsTag,
                                           true > ThisType;

   protected:

   typedef typename SuperentityTraits::StorageArrayType              StorageArrayType;
   typedef typename SuperentityTraits::GlobalIndexType               GlobalIndexType;
   typedef typename SuperentityTraits::LocalIndexType                LocalIndexType;

   typedef typename SuperentityTraits::StorageNetworkType   StorageNetworkType;
 
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

   void print( std::ostream& str ) const{}

   bool operator==( const ThisType& layer  ) const
   {
      return true;
   }

   StorageArrayType& getSuperentitiesIndices(){}

   const StorageArrayType& getSuperentitiesIndices() const{}

   bool save( tnlFile& file ) const
   {
      return true;
   }

   bool load( tnlFile& file )
   {
      return true;
   }
 
   typename tnlMeshTraits< MeshConfig >::GlobalIdArrayType& superentityIdsArray( DimensionsTag )
   {
      tnlAssert( false, );
      //return this->superentitiesIndices;
   }

   StorageNetworkType& getStorageNetwork( DimensionsTag )
   {
      tnlAssert( false, );
      //return this->storageNetwork;
   }

 
};

} // namespace TNL
