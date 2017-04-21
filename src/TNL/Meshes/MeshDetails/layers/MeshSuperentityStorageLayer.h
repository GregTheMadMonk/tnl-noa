/***************************************************************************
                          MeshSuperentityStorageLayer.h  -  description
                             -------------------
    begin                : Feb 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/File.h>
#include <TNL/Meshes/MeshDimensionTag.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/MeshDetails/traits/MeshSuperentityTraits.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag,
          bool SuperentityStorage =
             MeshSuperentityTraits< MeshConfig, EntityTopology, DimensionTag::value >::storageEnabled >
class MeshSuperentityStorageLayer;

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageLayers
   : public MeshSuperentityStorageLayer< MeshConfig,
                                            EntityTopology,
                                            MeshDimensionTag< MeshTraits< MeshConfig >::meshDimension > >
{
};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshSuperentityStorageLayer< MeshConfig, EntityTopology, DimensionTag, true >
   : public MeshSuperentityStorageLayer< MeshConfig, EntityTopology, typename DimensionTag::Decrement >
{
   typedef
      MeshSuperentityStorageLayer< MeshConfig, EntityTopology, typename DimensionTag::Decrement >  BaseType;

   static const int Dimension = DimensionTag::value;
   typedef MeshTraits< MeshConfig >                                                          MeshTraitsType;
   typedef typename MeshTraitsType::template SuperentityTraits< EntityTopology, Dimension > SuperentityTraitsType;

   protected:

   typedef typename SuperentityTraitsType::StorageArrayType       StorageArrayType;
   typedef typename SuperentityTraitsType::AccessArrayType        AccessArrayType;
   typedef typename SuperentityTraitsType::GlobalIndexType        GlobalIndexType;
   typedef typename SuperentityTraitsType::LocalIndexType         LocalIndexType;

   typedef typename SuperentityTraitsType::StorageNetworkType   StorageNetworkType;
 
   /****
     * Make visible setters and getters of the lower superentities
     */
    using BaseType::setNumberOfSuperentities;
    using BaseType::getNumberOfSuperentities;
    using BaseType::getSuperentityIndex;
    using BaseType::setSuperentityIndex;
    using BaseType::getSuperentitiesIndices;

    MeshSuperentityStorageLayer()
    {
    }

    /*~MeshSuperentityStorageLayer()
    {
       std::cerr << "      Destroying " << this->superentitiesIndices.getSize() << " superentities with "<< DimensionTag::value << " dimensions." << std::endl;
       std::cerr << "         this->superentitiesIndices.getName() = " << this->superentitiesIndices.getName() << std::endl;
       std::cerr << "         this->sharedSuperentitiesIndices.getName() = " << this->sharedSuperentitiesIndices.getName() << std::endl;
    }*/

    MeshSuperentityStorageLayer& operator = ( const MeshSuperentityStorageLayer& layer )
    {
       this->superentitiesIndices.setSize( layer.superentitiesIndices.getSize() );
       this->superentitiesIndices = layer.superentitiesIndices;
       this->sharedSuperentitiesIndices.bind( this->superentitiesIndices );
       return *this;
    }

    /****
     * Define setter/getter for the current level of the superentities
     */
    bool setNumberOfSuperentities( DimensionTag,
                                   const LocalIndexType size )
    {
       if( ! this->superentitiesIndices.setSize( size ) )
          return false;
       this->superentitiesIndices.setValue( -1 );
       this->sharedSuperentitiesIndices.bind( this->superentitiesIndices );
       return true;
    }

    LocalIndexType getNumberOfSuperentities( DimensionTag ) const
    {
       return this->superentitiesIndices.getSize();
    }

    void setSuperentityIndex( DimensionTag,
                              const LocalIndexType localIndex,
                              const GlobalIndexType globalIndex )
    {
       this->superentitiesIndices[ localIndex ] = globalIndex;
    }

    GlobalIndexType getSuperentityIndex( DimensionTag,
                                         const LocalIndexType localIndex ) const
    {
       return this->superentitiesIndices[ localIndex ];
    }

    AccessArrayType& getSuperentitiesIndices( DimensionTag )
    {
       return this->sharedSuperentitiesIndices;
    }

    const AccessArrayType& getSuperentitiesIndices( DimensionTag ) const
    {
       return this->sharedSuperentitiesIndices;
    }

    bool save( File& file ) const
    {
       if( ! BaseType::save( file ) ||
           ! this->superentitiesIndices.save( file ) )
       {
          //cerr << "Saving of the entity superentities layer with " << DimensionTag::value << " failed." << std::endl;
          return false;
       }
       return true;
    }

    bool load( File& file )
    {
       if( ! BaseType::load( file ) ||
           ! this->superentitiesIndices.load( file ) )
       {
          //cerr << "Loading of the entity superentities layer with " << DimensionTag::value << " failed." << std::endl;
          return false;
       }
       return true;
    }

    void print( std::ostream& str ) const
    {
       BaseType::print( str );
       str << std::endl << "\t Superentities with " << DimensionTag::value << " dimensions are: " << this->superentitiesIndices << ".";
    }

    bool operator==( const MeshSuperentityStorageLayer& layer  ) const
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
      typename MeshTraits< MeshConfig >::GlobalIdArrayType& superentityIdsArray( DimensionTag )
      {
         return this->superentitiesIndices;
      }
 
      using BaseType::getStorageNetwork;
      StorageNetworkType& getStorageNetwork( DimensionTag )
      {
         return this->storageNetwork;
      }
};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshSuperentityStorageLayer< MeshConfig, EntityTopology, DimensionTag, false >
   : public MeshSuperentityStorageLayer< MeshConfig, EntityTopology, typename DimensionTag::Decrement >
{
   public:

};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageLayer< MeshConfig, EntityTopology, MeshDimensionTag< EntityTopology::dimensions >, false >
{
   static const int Dimension = EntityTopology::dimensions;
   typedef MeshDimensionTag< EntityTopology::dimensions >        DimensionTag;

   typedef MeshSuperentityTraits< MeshConfig, EntityTopology, Dimension >      SuperentityTraits;

   typedef MeshSuperentityStorageLayer< MeshConfig,
                                           EntityTopology,
                                           DimensionTag,
                                           false > ThisType;

   protected:

   typedef typename SuperentityTraits::ContainerType              ContainerType;
   typedef typename ContainerType::ElementType                 GlobalIndexType;
   typedef int                                                 LocalIndexType;

   typedef typename SuperentityTraits::StorageNetworkType   StorageNetworkType;
 
   /****
    * These methods are due to 'using BaseType::...;' in the derived classes.
    */
   bool setNumberOfSuperentities( DimensionTag,
                                   const LocalIndexType size );
   LocalIndexType getNumberOfSuperentities( DimensionTag ) const;
   GlobalIndexType getSuperentityIndex( DimensionTag,
                                        const LocalIndexType localIndex ){}
   void setSuperentityIndex( DimensionTag,
                             const LocalIndexType localIndex,
                             const GlobalIndexType globalIndex ) {}

   void print( std::ostream& str ) const{}

   bool operator==( const ThisType& layer  ) const
   {
      return true;
   }

   ContainerType& getSuperentitiesIndices(){}

   const ContainerType& getSuperentitiesIndices() const{}

   bool save( File& file ) const
   {
      return true;
   }

   bool load( File& file )
   {
      return true;
   }
 
   typename MeshTraits< MeshConfig >::GlobalIdArrayType& superentityIdsArray( DimensionTag )
   {
      TNL_ASSERT( false, );
      //return this->superentitiesIndices;
   }

   StorageNetworkType& getStorageNetwork( DimensionTag )
   {
      TNL_ASSERT( false, );
     //return this->storageNetwork;
   }

};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageLayer< MeshConfig,
                                      EntityTopology,
                                      MeshDimensionTag< EntityTopology::dimensions >,
                                      true >
{
   static const int Dimension = EntityTopology::dimensions;
   typedef MeshDimensionTag< Dimension >                          DimensionTag;

   typedef MeshSuperentityTraits< MeshConfig,
                                     EntityTopology,
                                     Dimension >               SuperentityTraits;
   typedef MeshSuperentityStorageLayer< MeshConfig,
                                           EntityTopology,
                                           DimensionTag,
                                           true > ThisType;

   protected:

   typedef typename SuperentityTraits::StorageArrayType              StorageArrayType;
   typedef typename SuperentityTraits::GlobalIndexType               GlobalIndexType;
   typedef typename SuperentityTraits::LocalIndexType                LocalIndexType;

   typedef typename SuperentityTraits::StorageNetworkType   StorageNetworkType;
 
   /****
    * These methods are due to 'using BaseType::...;' in the derived classes.
    */
   bool setNumberOfSuperentities( DimensionTag,
                                   const LocalIndexType size );
   LocalIndexType getNumberOfSuperentities( DimensionTag ) const;
   GlobalIndexType getSuperentityIndex( DimensionTag,
                                        const LocalIndexType localIndex ){}
   void setSuperentityIndex( DimensionTag,
                             const LocalIndexType localIndex,
                             const GlobalIndexType globalIndex ) {}

   void print( std::ostream& str ) const{}

   bool operator==( const ThisType& layer  ) const
   {
      return true;
   }

   StorageArrayType& getSuperentitiesIndices(){}

   const StorageArrayType& getSuperentitiesIndices() const{}

   bool save( File& file ) const
   {
      return true;
   }

   bool load( File& file )
   {
      return true;
   }
 
   typename MeshTraits< MeshConfig >::GlobalIdArrayType& superentityIdsArray( DimensionTag )
   {
      TNL_ASSERT( false, );
      //return this->superentitiesIndices;
   }

   StorageNetworkType& getStorageNetwork( DimensionTag )
   {
      TNL_ASSERT( false, );
      //return this->storageNetwork;
   }

 
};

} // namespace Meshes
} // namespace TNL
