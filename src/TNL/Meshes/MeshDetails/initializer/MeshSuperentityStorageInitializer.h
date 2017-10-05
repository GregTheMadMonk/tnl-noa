/***************************************************************************
                          MeshSuperentityStorageInitializer.h  -  description
                             -------------------
    begin                : Feb 27, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <TNL/Meshes/MeshDimensionTag.h>
#include <algorithm>
#include <vector>

namespace TNL {
namespace Meshes {

template< typename MeshConfig >
class MeshInitializer;

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag,
          bool SuperentityStorage = MeshSuperentityTraits< MeshConfig, EntityTopology, DimensionTag::value >::storageEnabled >
class MeshSuperentityStorageInitializerLayer;

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageInitializer :
   public MeshSuperentityStorageInitializerLayer< MeshConfig, EntityTopology, MeshDimensionTag< MeshTraits< MeshConfig >::meshDimension > >
{};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshSuperentityStorageInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          DimensionTag,
                                          true >
   : public MeshSuperentityStorageInitializerLayer< MeshConfig,
                                                EntityTopology,
                                                typename DimensionTag::Decrement >
{
   typedef MeshSuperentityStorageInitializerLayer< MeshConfig,
                                                      EntityTopology,
                                                      typename DimensionTag::Decrement >      BaseType;

   static const int Dimension = DimensionTag::value;
   typedef MeshDimensionTag< EntityTopology::dimensions >                                       EntityDimension;
	
   typedef MeshTraits< MeshConfig >                                                             MeshTraitsType;
   typedef typename MeshTraitsType::GlobalIdArrayType                                           GlobalIdArrayType; 
   typedef typename MeshTraitsType::GlobalIndexType                                             GlobalIndexType;
   typedef typename MeshTraitsType::LocalIndexType                                              LocalIndexType;
   typedef MeshInitializer< MeshConfig >                                                        MeshInitializerType;
   typedef typename MeshTraitsType::template SuperentityTraits< EntityTopology, Dimension >    SuperentityTraitsType;
   typedef typename SuperentityTraitsType::StorageNetworkType                                   SuperentityStorageNetwork;

   public:
      using BaseType::addSuperentity;
	
      void addSuperentity( DimensionTag, GlobalIndexType entityIndex, GlobalIndexType superentityIndex)
      {
         //cout << "Adding superentity with " << DimensionTag::value << " dimensions of enity with " << EntityDimension::value << " ... " << std::endl;
         indexPairs.push_back( IndexPair{ entityIndex, superentityIndex } );
      }

      using BaseType::initSuperentities;
      void initSuperentities( MeshInitializerType& meshInitializer )
      {
         throw( 0 ); // TODO: fix this - or it may work with newer version of gcc
         /*std::sort( indexPairs.begin(),
                    indexPairs.end(),
                    []( IndexPair pair0, IndexPair pair1 ){ return ( pair0.entityIndex < pair1.entityIndex ); } );*/

         GlobalIdArrayType &superentityIdsArray = meshInitializer.template meshSuperentityIdsArray< EntityDimension, DimensionTag >();
         superentityIdsArray.setSize( static_cast< GlobalIndexType >( indexPairs.size() )  );
         GlobalIndexType currentBegin = 0;
         GlobalIndexType lastEntityIndex = 0;
        std::cout << "There are " << superentityIdsArray.getSize() << " superentities with " << DimensionTag::value << " dimensions of enities with " << EntityDimension::value << " ... " << std::endl;
         for( GlobalIndexType i = 0; i < superentityIdsArray.getSize(); i++)
         {
            superentityIdsArray[ i ] = indexPairs[i].superentityIndex;
 
            //cout << "Adding superentity " << indexPairs[i].superentityIndex << " to entity " << lastEntityIndex << std::endl;
            if( indexPairs[ i ].entityIndex != lastEntityIndex )
            {
               meshInitializer.template superentityIdsArray< DimensionTag >( meshInitializer.template meshEntitiesArray< EntityDimension >()[ lastEntityIndex ] ).bind( superentityIdsArray, currentBegin, i - currentBegin );
               currentBegin = i;
               lastEntityIndex = indexPairs[ i ].entityIndex;
            }
         }

         meshInitializer.template superentityIdsArray< DimensionTag >( meshInitializer.template meshEntitiesArray< EntityDimension >()[ lastEntityIndex ] ).bind( superentityIdsArray, currentBegin, superentityIdsArray.getSize() - currentBegin );
         indexPairs.clear();
 
         /****
          * Network initializer
          */
         SuperentityStorageNetwork& superentityStorageNetwork = meshInitializer.template meshSuperentityStorageNetwork< EntityTopology, DimensionTag >();
         //GlobalIndexType lastEntityIndex( 0 );
         superentityStorageNetwork.setRanges(
            meshInitializer.template meshEntitiesArray< EntityDimension >().getSize(),
            meshInitializer.template meshEntitiesArray< DimensionTag >().getSize() );
         lastEntityIndex = 0;
         typename SuperentityStorageNetwork::ValuesAllocationVectorType storageNetworkAllocationVector;
         storageNetworkAllocationVector.setSize( meshInitializer.template meshEntitiesArray< EntityDimension >().getSize() );
         storageNetworkAllocationVector.setValue( 0 );
         for( GlobalIndexType i = 0; i < superentityIdsArray.getSize(); i++)
         {
            if( indexPairs[ i ].entityIndex == lastEntityIndex )
               storageNetworkAllocationVector[ lastEntityIndex ]++;
            else
               lastEntityIndex++;
         }
         superentityStorageNetwork.allocate( storageNetworkAllocationVector );
         lastEntityIndex = 0;
         LocalIndexType superentitiesCount( 0 );
         typename SuperentityStorageNetwork::ValuesAccessorType superentitiesIndecis =
            superentityStorageNetwork.getValues( lastEntityIndex );
         for( GlobalIndexType i = 0; i < superentityIdsArray.getSize(); i++)
         {
            if( indexPairs[ i ].entityIndex != lastEntityIndex )
            {
               superentitiesIndecis = superentityStorageNetwork.getValues( ++lastEntityIndex );
               superentitiesCount = 0;
            }
            superentitiesIndecis[ superentitiesCount++ ] =  indexPairs[ i ].superentityIndex;
         }
         BaseType::initSuperentities( meshInitializer );
      }

   private:
      struct IndexPair
      {
         GlobalIndexType entityIndex;
         GlobalIndexType superentityIndex;
      };

      std::vector< IndexPair > indexPairs;
 
};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshSuperentityStorageInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          DimensionTag,
                                          false >
   : public MeshSuperentityStorageInitializerLayer< MeshConfig,
                                                EntityTopology,
                                                typename DimensionTag::Decrement >
{
   typedef MeshSuperentityStorageInitializerLayer< MeshConfig,
                                                EntityTopology,
                                                typename DimensionTag::Decrement > BaseType;
   typedef MeshInitializer< MeshConfig >                                      MeshInitializerType;
 
   public:
   void addSuperentity()                           {} // This method is due to 'using BaseType::...;' in the derived classes.
   using BaseType::initSuperentities;
   void initSuperentities( MeshInitializerType& ) { std::cerr << "***" << std::endl;}
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          MeshDimensionTag< EntityTopology::dimensions >,
                                          true >
{
   typedef MeshInitializer< MeshConfig >                                      MeshInitializerType;
 
   public:
   void addSuperentity()                           {} // This method is due to 'using BaseType::...;' in the derived classes.
   void initSuperentities( MeshInitializerType& ) {}
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          MeshDimensionTag< EntityTopology::dimensions >,
                                          false >
{
   typedef MeshInitializer< MeshConfig >                                      MeshInitializerType;

   public:
   void addSuperentity()                           {} // This method is due to 'using BaseType::...;' in the derived classes.
   void initSuperentities( MeshInitializerType& ) {}
};

} // namespace Meshes
} // namespace TNL
